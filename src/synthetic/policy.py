import numpy as np
from scipy.stats import rankdata


def gen_eps_greedy(
    expected_reward: np.ndarray,
    is_optimal: bool = True,
    k: int = 1,
    eps: float = 0.0,
) -> np.ndarray:
    "Generate an evaluation policy via the epsilon-greedy rule."
    if is_optimal:
        rank = rankdata(-expected_reward, axis=1)
    else:
        rank = rankdata(expected_reward, axis=1)
    is_topk = rank <= k
    action_dist = ((1.0 - eps) / k) * is_topk
    action_dist += eps / expected_reward.shape[1]
    action_dist /= action_dist.sum(1)[:, np.newaxis]

    return action_dist[:, :, np.newaxis]
