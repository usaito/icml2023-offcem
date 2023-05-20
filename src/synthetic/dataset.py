from dataclasses import dataclass
import enum

import numpy as np
from obp.dataset import linear_behavior_policy
from obp.dataset import linear_reward_function
from obp.dataset import polynomial_reward_function
from obp.dataset import SyntheticBanditDatasetWithActionEmbeds
from obp.types import BanditFeedback
from obp.utils import sample_action_fast
from obp.utils import softmax
from scipy.stats import rankdata
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1 + np.exp(-x))


class RewardType(enum.Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"

    def __repr__(self) -> str:

        return str(self)


def cluster_effect_function(
    context: np.ndarray,
    cluster_context: np.ndarray,
    random_state: int,
) -> np.ndarray:
    g_x_e = polynomial_reward_function(
        context=context,
        action_context=cluster_context,
        random_state=random_state,
    )
    random_ = check_random_state(random_state)
    (a, b, c, d) = random_.uniform(-3, 3, size=4)
    x_a = 1 / context[:, :3].mean(axis=1)
    x_b = 1 / context[:, 2:8].mean(axis=1)
    x_c = context[:, 1:3].mean(axis=1)
    x_d = context[:, 5:].mean(axis=1)
    g_x_e += a * (x_a[:, np.newaxis] < 1.5)
    g_x_e += b * (x_b[:, np.newaxis] < -0.5)
    g_x_e += c * (x_c[:, np.newaxis] > 3.0)
    g_x_e += d * (x_d[:, np.newaxis] < 1.0)

    return g_x_e


@dataclass
class SyntheticBanditDataset(SyntheticBanditDatasetWithActionEmbeds):
    def obtain_batch_bandit_feedback(
        self, n_rounds: int, n_users: int = 200, n_clusters: int = 30
    ) -> BanditFeedback:
        check_scalar(n_rounds, "n_rounds", int, min_val=1)
        fixed_user_contexts = self.random_.normal(size=(n_users, self.dim_context))
        user_idx = self.random_.choice(n_users, size=n_rounds)

        # define (near-deterministic) action embeddings
        self.p_e_a = softmax(
            self.random_.normal(
                scale=4, size=(self.n_actions, self.n_cat_per_dim, self.n_cat_dim)
            ),
        )
        action_embed = np.zeros((self.n_actions, self.n_cat_dim), dtype=int)
        for d in np.arange(self.n_cat_dim):
            action_embed[:, d] = sample_action_fast(
                self.p_e_a[:, :, d],
                random_state=d,
            )
        action_context = action_embed[:, self.n_unobserved_cat_dim :]
        action_context_one_hot = OneHotEncoder(
            drop="first", sparse=False
        ).fit_transform(action_context)

        # generate action clusters
        cluster_logits = linear_behavior_policy(
            context=action_context_one_hot,
            action_context=np.eye(n_clusters),
        )
        clusters = sample_action_fast(softmax(cluster_logits / 10))
        n_clusters = np.unique(clusters).shape[0]
        clusters = rankdata(-clusters, method="dense") - 1

        # calc expected rewards given context and action (n_data, n_actions)
        fixed_q_x_a = np.zeros((n_users, self.n_actions))
        g_x_e = cluster_effect_function(
            context=fixed_user_contexts,
            cluster_context=np.eye(n_clusters),
            random_state=self.random_state,
        )
        for c in np.arange(n_clusters):
            fixed_q_x_a[:, clusters == c] = linear_reward_function(
                context=fixed_user_contexts,
                action_context=action_context_one_hot[clusters == c],
                random_state=self.random_state + c,
            )
            fixed_q_x_a[:, clusters == c] += g_x_e[:, c][:, np.newaxis]

        contexts = fixed_user_contexts[user_idx]
        q_x_a = fixed_q_x_a[user_idx]

        # sample actions for each round based on the behavior policy
        if self.behavior_policy_function is None:
            if RewardType(self.reward_type) == RewardType.BINARY:
                pi_b_logits = sigmoid(q_x_a)
            else:
                pi_b_logits = q_x_a
        else:
            pi_b_logits = self.behavior_policy_function(
                context=contexts,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        if self.n_deficient_actions > 0:
            pi_b = np.zeros_like(q_x_a)
            n_supported_actions = self.n_actions - self.n_deficient_actions
            supported_actions = np.argsort(
                self.random_.gumbel(size=(n_rounds, self.n_actions)), axis=1
            )[:, ::-1][:, :n_supported_actions]
            supported_actions_idx = (
                np.tile(np.arange(n_rounds), (n_supported_actions, 1)).T,
                supported_actions,
            )
            pi_b[supported_actions_idx] = softmax(
                self.beta * pi_b_logits[supported_actions_idx]
            )
        else:
            pi_b = softmax(self.beta * pi_b_logits)
        actions = sample_action_fast(pi_b, random_state=self.random_state)

        # sample rewards given the context and action embeddings
        if RewardType(self.reward_type) == RewardType.BINARY:
            q_x_a = sigmoid(q_x_a)
            expected_rewards_factual = q_x_a[np.arange(actions.shape[0]), actions]
            rewards = self.random_.binomial(n=1, p=expected_rewards_factual)
        elif RewardType(self.reward_type) == RewardType.CONTINUOUS:
            expected_rewards_factual = q_x_a[np.arange(actions.shape[0]), actions]
            rewards = self.random_.normal(
                loc=expected_rewards_factual, scale=self.reward_std, size=n_rounds
            )
        reward_mat = np.zeros((n_users, self.n_actions))
        for u, a, r in zip(user_idx, actions, rewards):
            reward_mat[u, a] = r

        clusters_ = np.zeros((self.n_actions, n_clusters))
        clusters_[np.arange(self.n_actions), clusters] = 1
        clusters_ = np.tile(clusters_, (n_users, 1, 1))

        return dict(
            n_rounds=n_rounds,
            n_users=n_users,
            n_actions=self.n_actions,
            clusters=clusters_,
            action_context=action_context,
            action_context_one_hot=action_context_one_hot,
            user_idx=user_idx,
            fixed_user_contexts=fixed_user_contexts,
            fixed_expected_rewards=fixed_q_x_a,
            context=contexts,
            action=actions,
            position=None,
            reward=rewards,
            reward_mat=reward_mat,
            obs_mat=(reward_mat != 0).astype(int),
            expected_reward=q_x_a,
            g_x_e=g_x_e,
            p_e_a=self.p_e_a[:, :, self.n_unobserved_cat_dim :],
            action_embed=action_context[actions],
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pi_b[np.arange(n_rounds), actions],
        )
