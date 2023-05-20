from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Tuple

import numpy as np
from obp.dataset import BaseRealBanditDataset
from obp.utils import check_array
from obp.utils import sample_action_fast
from obp.utils import softmax
from scipy import sparse
from scipy.sparse.coo import coo_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class ExtremeBanditDataset(BaseRealBanditDataset):
    n_components: int = 100
    reward_std: float = 1.0
    max_reward_noise: float = 0.2
    dataset_name: str = "eurlex"  # wiki or eurlex
    random_state: int = 12345

    def __post_init__(self):
        self.data_path = Path().cwd().parents[1] / "data" / self.dataset_name
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.sc = StandardScaler()
        if self.dataset_name == "eurlex":
            self.min_label_frequency = 1
        elif self.dataset_name == "wiki":
            self.min_label_frequency = 9
        self.random_ = check_random_state(self.random_state)
        self.load_raw_data()

        # generate reward_noise (depends on each action)
        self.eta = self.random_.uniform(
            0, self.max_reward_noise, size=(1, self.train_label.shape[1])
        )

        # train a classifier to define a logging policy
        self.train_pi_b()

    def load_raw_data(self) -> None:
        """Load raw dataset."""
        self.train_data, self.train_label = self.pre_process(
            self.data_path / "train.txt"
        )
        self.test_data, self.test_label = self.pre_process(self.data_path / "test.txt")
        self.n_train, self.n_test = self.train_data.shape[0], self.test_data.shape[0]
        # delete some rare actions
        all_label = (
            sparse.vstack([self.train_label, self.test_label]).astype(np.int8).toarray()
        )
        idx = all_label.sum(axis=0) >= self.min_label_frequency
        all_label = all_label[:, idx]
        self.n_actions = all_label.shape[1]
        self.train_label = sparse.csr_matrix(
            all_label[: self.n_train], dtype=np.float32
        ).toarray()
        self.test_label = sparse.csr_matrix(
            all_label[self.n_train :], dtype=np.float32
        ).toarray()

    def pre_process(
        self, file_path: Path
    ) -> Tuple[int, int, int, coo_matrix, coo_matrix]:
        """Preprocess raw dataset."""
        data_file = open(file_path, "r")
        num_data, num_feat, num_label = data_file.readline().split(" ")
        num_data, num_feat, num_label = int(num_data), int(num_feat), int(num_label)

        data, label = [], []
        for i in range(num_data):
            raw_data_i = data_file.readline().split(" ")
            label_pos_i = [int(x) for x in raw_data_i[0].split(",") if x != ""]
            data_pos_i = [int(x.split(":")[0]) for x in raw_data_i[1:]]
            data_i = [float(x.split(":")[1]) for x in raw_data_i[1:]]
            label.append(
                sparse.csr_matrix(
                    ([1.0] * len(label_pos_i), label_pos_i, [0, len(label_pos_i)]),
                    shape=(1, num_label),
                )
            )
            data.append(
                sparse.csr_matrix(
                    (data_i, data_pos_i, [0, len(data_i)]), shape=(1, num_feat)
                )
            )
        return sparse.vstack(data).toarray(), sparse.vstack(label).toarray()

    def train_pi_b(
        self,
        n_rounds: int = 2000,
    ) -> None:
        idx = self.random_.choice(self.n_test, size=n_rounds, replace=False)
        contexts = self.test_data[idx]
        contexts = self.sc.fit_transform(self.pca.fit_transform(contexts))
        expected_rewards = self.test_label[idx]
        expected_rewards = expected_rewards * (1 - self.eta)
        expected_rewards += (1 - expected_rewards) * (self.eta - 1)
        expected_rewards = sigmoid(expected_rewards)

        self.regressor = MultiOutputRegressor(Ridge(max_iter=500, random_state=12345))
        self.regressor.fit(contexts, expected_rewards)

    def compute_pi_b(
        self,
        train_contexts: np.ndarray,
        train_expected_rewards: np.ndarray,
        beta: float = 1.0,
    ) -> np.ndarray:
        r_hat = self.regressor.predict(train_contexts)
        pi_b = softmax(r_hat * beta)
        return pi_b

    def obtain_batch_bandit_feedback(
        self, n_rounds: Optional[int] = None, n_users: int = 300, beta: float = 10
    ) -> dict:
        """Obtain batch logged bandit data."""
        if n_rounds is None:
            n_rounds = self.n_train
        idx = self.random_.choice(self.n_train, size=n_rounds, replace=False)
        fixed_user_contexts = self.train_data[idx]
        fixed_user_contexts = self.sc.fit_transform(
            self.pca.fit_transform(fixed_user_contexts)
        )
        fixed_q_x_a = self.train_label[idx]
        fixed_q_x_a = fixed_q_x_a * (1 - self.eta)
        fixed_q_x_a += (1 - fixed_q_x_a) * (self.eta - 1)
        fixed_q_x_a = sigmoid(fixed_q_x_a)

        user_idx = self.random_.choice(n_users, size=n_rounds)
        contexts = fixed_user_contexts[user_idx]
        q_x_a = fixed_q_x_a[user_idx]
        pi_b = self.compute_pi_b(contexts, q_x_a, beta=beta)
        actions = sample_action_fast(pi_b, random_state=self.random_state)
        q_x_a_factual = q_x_a[np.arange(n_rounds), actions]
        rewards = self.random_.binomial(n=1, p=q_x_a_factual)

        reward_mat = np.zeros((n_users, self.n_actions))
        for u, a, r in zip(user_idx, actions, rewards):
            reward_mat[u, a] = r

        return dict(
            n_rounds=n_rounds,
            n_users=n_users,
            user_idx=user_idx,
            n_actions=self.n_actions,
            action_context=np.eye(self.n_actions),
            context=contexts,
            fixed_user_contexts=fixed_user_contexts,
            action=actions,
            position=None,
            reward=rewards,
            reward_mat=reward_mat,
            obs_mat=(reward_mat != 0).astype(int),
            expected_reward=fixed_q_x_a,
            q_x_a=q_x_a,
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pi_b[np.arange(n_rounds), actions],
        )

    @staticmethod
    def calc_ground_truth_policy_value(
        expected_reward: np.ndarray, action_dist: np.ndarray
    ) -> float:
        check_array(array=expected_reward, name="expected_reward", expected_dim=2)
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        if expected_reward.shape[0] != action_dist.shape[0]:
            raise ValueError(
                "Expected `expected_reward.shape[0] = action_dist.shape[0]`, but found it False"
            )
        if expected_reward.shape[1] != action_dist.shape[1]:
            raise ValueError(
                "Expected `expected_reward.shape[1] = action_dist.shape[1]`, but found it False"
            )

        return np.average(expected_reward, weights=action_dist[:, :, 0], axis=1).mean()
