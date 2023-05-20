from dataclasses import dataclass
from itertools import permutations
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import MarginalizedInverseProbabilityWeighting as MIPS
from obp.ope import OffPolicyEvaluation
from obp.ope import RegressionModel
from obp.utils import check_array
from obp.utils import check_ope_inputs
from scipy import stats
from scipy.stats import rankdata
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.utils import check_random_state
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR


class PairWiseRegression(nn.Module):
    def __init__(
        self,
        n_actions: int,
        x_dim: int,
        hidden_dim: int = 30,
    ):
        super(PairWiseRegression, self).__init__()
        # init
        self.n_actions = n_actions
        self.x_dim = x_dim

        # relative reward network
        self.fc1 = nn.Linear(self.x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.n_actions)

    def rel_reward_pred(self, x):
        h_hat = F.elu(self.fc1(x))
        h_hat = F.elu(self.fc2(h_hat))
        h_hat = self.fc3(h_hat)
        return h_hat

    def forward(self, x, a1, a2, r1, r2):
        h_hat = self.rel_reward_pred(x)
        h_hat1, h_hat2 = h_hat[:, a1], h_hat[:, a2]
        loss = ((r1 - r2) - (h_hat1 - h_hat2)) ** 2

        return loss.mean()

    def predict(self, x, fixed_action_context):
        with torch.no_grad():
            user_emb = F.elu(self.fc1_user(x))
            user_emb = F.elu(self.fc2_user(user_emb))
            user_emb = self.fc3_user(user_emb).unsqueeze(dim=-1)
            item_emb = F.elu(self.fc1_item(fixed_action_context))
            item_emb = F.elu(self.fc2_item(item_emb))
            item_emb = self.fc3_item(item_emb).view(self.n_actions, -1)
            h_hat = user_emb.matmul(item_emb.T)
            return h_hat.detach().numpy()


@dataclass
class PairWiseDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action1: np.ndarray
    action2: np.ndarray
    reward1: np.ndarray
    reward2: np.ndarray

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action1[index],
            self.action2[index],
            self.reward1[index],
            self.reward2[index],
        )

    def __len__(self):
        return self.context.shape[0]


def make_pairwise_data(bandit_data: dict):
    n_users = bandit_data["n_users"]
    fixed_user_contexts = bandit_data["fixed_user_contexts"]
    action_set = np.arange(bandit_data["n_actions"])
    obs_mat = bandit_data["obs_mat"]
    reward_mat = bandit_data["reward_mat"]

    contexts_ = [fixed_user_contexts[0]]
    actions1_ = [0]
    actions2_ = [0]
    rewards1_ = [0]
    rewards2_ = [0]
    for u in np.arange(n_users):
        obs_actions = action_set[obs_mat[u, :] == 1]
        for (a1, a2) in permutations(obs_actions, 2):
            r1, r2 = reward_mat[u, (a1, a2)]
            contexts_.append(fixed_user_contexts[u])
            actions1_.append(a1), actions2_.append(a2)
            rewards1_.append(r1), rewards2_.append(r2)

    return PairWiseDataset(
        torch.from_numpy(np.array(contexts_)).float(),
        torch.from_numpy(np.array(actions1_)).long(),
        torch.from_numpy(np.array(actions2_)).long(),
        torch.from_numpy(np.array(rewards1_)).float(),
        torch.from_numpy(np.array(rewards2_)).float(),
    )


def train_pairwise_model(
    bandit_data: dict,
    lr: float = 1e-2,
    batch_size: int = 128,
    num_epochs: int = 30,
    gamma: float = 0.95,
    weight_decay: float = 1e-4,
    random_state: int = 12345,
    verbose: bool = False,
) -> None:
    pairwise_dataset = make_pairwise_data(bandit_data)
    data_loader = torch.utils.data.DataLoader(
        pairwise_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    model = PairWiseRegression(
        n_actions=bandit_data["n_actions"],
        x_dim=bandit_data["context"].shape[1],
    )
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    model.train()
    loss_list = []
    for _ in range(num_epochs):
        losses = []
        for x, a1, a2, r1, r2 in data_loader:
            loss = model(x, a1, a2, r1, r2)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
        if verbose:
            print(_, np.average(losses))
        loss_list.append(np.average(losses))
        scheduler.step()
    x = torch.from_numpy(bandit_data["context"]).float()
    h_hat_mat = model.rel_reward_pred(x).detach().numpy()

    return h_hat_mat


def train_reward_model_via_two_stage(
    bandit_data: dict,
    clusters: np.ndarray,
    need_q_x_a: bool = True,
    random_state: int = 12345,
) -> np.ndarray:

    ### two-step reward regression for OffCEM ###
    ref_mat = np.tile(np.arange(clusters.shape[-1]), (clusters.shape[1], 1))
    cluster_idx_mat = np.zeros_like(clusters[:, :, 0]).astype(int)
    for i, clusters_ in enumerate(clusters):
        cluster_idx_mat[i] = (ref_mat * clusters_).sum(1)
    cluster_idx_mat = rankdata(cluster_idx_mat, method="dense", axis=1) - 1
    n_clusters = np.unique(cluster_idx_mat).shape[0]

    h_hat_mat = train_pairwise_model(bandit_data)
    reward_residual = bandit_data["reward"].astype(float)
    reward_residual -= h_hat_mat[
        np.arange(bandit_data["context"].shape[0]), bandit_data["action"]
    ]

    reg_model = RegressionModel(
        n_actions=n_clusters,
        base_model=MLP(hidden_layer_sizes=(30, 30, 30), random_state=random_state),
    )
    observed_cluster = cluster_idx_mat[bandit_data["user_idx"], bandit_data["action"]]
    g_hat_mat = reg_model.fit_predict(
        context=bandit_data["context"],
        action=observed_cluster,
        reward=reward_residual,
    )[:, :, 0]

    f_x_a_e = h_hat_mat
    cluster_idx_mat_ = cluster_idx_mat[bandit_data["user_idx"]]
    for i in np.arange(f_x_a_e.shape[0]):
        f_x_a_e[i] += g_hat_mat[i][cluster_idx_mat_[i]]
    f_x_a_e = f_x_a_e[:, :, np.newaxis]

    if need_q_x_a:
        ### one-step reward regression ###
        reg_model = RegressionModel(
            n_actions=bandit_data["n_actions"],
            action_context=bandit_data["action_context"],
            base_model=MLPClassifier(
                hidden_layer_sizes=(30, 30, 30),
                random_state=random_state,
            ),
        )
        q_x_a = reg_model.fit_predict(
            context=bandit_data["context"],
            action=bandit_data["action"],
            reward=bandit_data["reward"],
        )

        return f_x_a_e, q_x_a

    else:
        return f_x_a_e


def train_clustering(
    bandit_data: dict,
    n_clusters: int = 100,
    random_state: int = 12345,
):
    n_rounds = bandit_data["n_rounds"]
    n_actions = bandit_data["n_actions"]
    random_ = check_random_state(random_state)

    clusters = random_.choice(n_clusters, size=n_actions)
    clusters_ = np.zeros((n_actions, n_clusters))
    clusters_[np.arange(n_actions), clusters] = 1

    return np.tile(clusters_, (n_rounds, 1, 1))


@dataclass
class OffCEM(MIPS):
    """The OffCEM Estimator."""

    def _estimate_round_rewards(
        self,
        context: np.ndarray,
        reward: np.ndarray,
        action: np.ndarray,
        action_embed: np.ndarray,
        pi_b: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        p_e_a: Optional[np.ndarray] = None,
        with_dev: bool = False,
        **kwargs,
    ) -> np.ndarray:
        n = reward.shape[0]
        if p_e_a is not None:
            observed_clusters = p_e_a[np.arange(n), action, :]
            pi_b_c = np.zeros((p_e_a.shape[0], p_e_a.shape[2]))
            pi_e_c = np.zeros((p_e_a.shape[0], p_e_a.shape[2]))
            for i in range(n):
                pi_b_c[i] = pi_b[i, :, 0] @ p_e_a[i]
                pi_e_c[i] = action_dist[i, :, 0] @ p_e_a[i]
            pi_b_c_ = (pi_b_c * observed_clusters).sum(1)
            pi_e_c_ = (pi_e_c * observed_clusters).sum(1)
            w_x_e = pi_e_c_ / pi_b_c_

        else:
            w_x_e = self._estimate_w_x_e(
                context=context,
                action=action,
                action_embed=action_embed,
                pi_e=action_dist[np.arange(n), :, position],
                pi_b=pi_b[np.arange(n), :, position],
            )
            self.max_w_x_e = w_x_e.max()

        if with_dev:
            r_hat = reward * w_x_e
            cnf = np.sqrt(np.var(r_hat) / (n - 1))
            cnf *= stats.t.ppf(1.0 - (self.delta / 2), n - 1)

            return r_hat.mean(), cnf

        q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n), :, position]
        q_hat_factual = estimated_rewards_by_reg_model[np.arange(n), action, position]
        pi_e_at_position = action_dist[np.arange(n), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += w_x_e * (reward - q_hat_factual)

        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_embed: np.ndarray,
        pi_b: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        context: Optional[np.ndarray] = None,
        p_e_a: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action_embed, name="action_embed", expected_dim=2)
        check_array(array=pi_b, name="pi_b", expected_dim=3)
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        check_ope_inputs(
            action_dist=pi_b,
            position=position,
            action=action,
            reward=reward,
        )
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        if p_e_a is not None:
            check_array(array=p_e_a, name="p_e_a", expected_dim=3)
        else:
            check_array(array=context, name="context", expected_dim=2)

        if action_embed.shape[1] > 1 and self.embedding_selection_method is not None:
            if self.embedding_selection_method == "exact":
                return self._estimate_with_exact_pruning(
                    context=context,
                    reward=reward,
                    action=action,
                    action_embed=action_embed,
                    position=position,
                    pi_b=pi_b,
                    action_dist=action_dist,
                    p_e_a=p_e_a,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                )
            elif self.embedding_selection_method == "greedy":
                return self._estimate_with_greedy_pruning(
                    context=context,
                    reward=reward,
                    action=action,
                    action_embed=action_embed,
                    position=position,
                    pi_b=pi_b,
                    action_dist=action_dist,
                    p_e_a=p_e_a,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                )
        else:
            return self._estimate_round_rewards(
                context=context,
                reward=reward,
                action=action,
                action_embed=action_embed,
                position=position,
                pi_b=pi_b,
                action_dist=action_dist,
                p_e_a=p_e_a,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            ).mean()


@dataclass
class OffPolicyEvaluation(OffPolicyEvaluation):
    def _create_estimator_inputs(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_importance_weights: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        action_embed: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        pi_b: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        p_e_a: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Create input dictionary to estimate policy value using subclasses of `BaseOffPolicyEstimator`"""
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        if estimated_rewards_by_reg_model is None:
            pass
        elif isinstance(estimated_rewards_by_reg_model, dict):
            for estimator_name, value in estimated_rewards_by_reg_model.items():
                check_array(
                    array=value,
                    name=f"estimated_rewards_by_reg_model[{estimator_name}]",
                    expected_dim=3,
                )
                if value.shape != action_dist.shape:
                    raise ValueError(
                        f"Expected `estimated_rewards_by_reg_model[{estimator_name}].shape == action_dist.shape`, but found it False."
                    )
        else:
            check_array(
                array=estimated_rewards_by_reg_model,
                name="estimated_rewards_by_reg_model",
                expected_dim=3,
            )
            if estimated_rewards_by_reg_model.shape != action_dist.shape:
                raise ValueError(
                    "Expected `estimated_rewards_by_reg_model.shape == action_dist.shape`, but found it False"
                )
        for var_name, value_or_dict in {
            "estimated_pscore": estimated_pscore,
            "estimated_importance_weights": estimated_importance_weights,
            "action_embed": action_embed,
            "pi_b": pi_b,
            "p_e_a": p_e_a,
        }.items():
            if value_or_dict is None:
                pass
            elif isinstance(value_or_dict, dict):
                for estimator_name, value in value_or_dict.items():
                    expected_dim = 1
                    if var_name in ["p_e_a", "pi_b"]:
                        expected_dim = 3
                    elif var_name in ["action_embed"]:
                        expected_dim = 2
                    check_array(
                        array=value,
                        name=f"{var_name}[{estimator_name}]",
                        expected_dim=expected_dim,
                    )
            else:
                expected_dim = 1
                if var_name in ["p_e_a", "pi_b"]:
                    expected_dim = 3
                elif var_name in ["action_embed"]:
                    expected_dim = 2
                check_array(
                    array=value_or_dict, name=var_name, expected_dim=expected_dim
                )

        estimator_inputs = {
            estimator_name: {
                input_: self.bandit_feedback[input_]
                for input_ in ["action", "position", "reward", "context"]
            }
            for estimator_name in self.ope_estimators_
        }

        for estimator_name in self.ope_estimators_:
            if "pscore" in self.bandit_feedback:
                estimator_inputs[estimator_name]["pscore"] = self.bandit_feedback[
                    "pscore"
                ]
            else:
                estimator_inputs[estimator_name]["pscore"] = None
            estimator_inputs[estimator_name]["action_dist"] = action_dist
            estimator_inputs = self._preprocess_model_based_input(
                estimator_inputs=estimator_inputs,
                estimator_name=estimator_name,
                model_based_input={
                    "estimated_rewards_by_reg_model": estimated_rewards_by_reg_model,
                    "estimated_pscore": estimated_pscore,
                    "estimated_importance_weights": estimated_importance_weights,
                    "action_embed": action_embed,
                    "pi_b": pi_b,
                    "p_e_a": p_e_a,
                },
            )
        return estimator_inputs


def run_ope(
    bandit_data: Dict,
    pi_e: np.ndarray,
    action_clusters: np.ndarray,
    f_x_a: np.ndarray,
    q_x_a: np.ndarray,
) -> np.ndarray:

    n_actions = bandit_data["n_actions"]
    ope_estimators = [
        IPS(estimator_name="IPS"),
        DR(estimator_name="DR"),
        DM(estimator_name="DM"),
    ]
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_data,
        ope_estimators=ope_estimators,
    )
    estimated_policy_values = ope.estimate_policy_values(
        action_dist=pi_e,
        estimated_rewards_by_reg_model=q_x_a,
        pi_b=bandit_data["pi_b"],
    )

    ref_mat = np.tile(
        np.arange(action_clusters.shape[-1]), (action_clusters.shape[1], 1)
    )
    cluster_idx_mat = np.zeros_like(action_clusters[:, :, 0]).astype(int)
    for i, clusters_ in enumerate(action_clusters):
        cluster_idx_mat[i] = (ref_mat * clusters_).sum(1)
    cluster_idx_mat = rankdata(cluster_idx_mat, method="dense", axis=1) - 1
    action_embed = cluster_idx_mat[
        np.arange(cluster_idx_mat.shape[0]), bandit_data["action"]
    ]

    ope_estimators_with_clustering = [
        OffCEM(
            n_actions=n_actions,
            estimator_name="+clustering",
        ),
        OffCEM(
            n_actions=n_actions,
            estimator_name="OffCEM",
        ),
    ]
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_data,
        ope_estimators=ope_estimators_with_clustering,
    )
    estimated_policy_values_with_clustering = ope.estimate_policy_values(
        action_dist=pi_e,
        action_embed=action_embed[:, np.newaxis],
        pi_b=bandit_data["pi_b"],
        p_e_a={
            "+clustering": action_clusters,
            "OffCEM": action_clusters,
        },
        estimated_rewards_by_reg_model={
            "+clustering": np.zeros_like(q_x_a),
            "OffCEM": f_x_a,
        },
    )
    estimated_policy_values.update(estimated_policy_values_with_clustering)

    return estimated_policy_values
