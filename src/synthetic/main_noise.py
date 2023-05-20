from logging import getLogger
from pathlib import Path
from time import time
import warnings

from dataset import SyntheticBanditDataset
import hydra
import numpy as np
from obp.dataset import linear_reward_function
from omegaconf import DictConfig
from ope import run_ope
from ope import train_reward_model_via_two_stage
import pandas as pd
from pandas import DataFrame
from policy import gen_eps_greedy
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = getLogger(__name__)


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
    logger.info(f"The current working directory is {Path().cwd()}")
    start_time = time()

    # log path
    log_path = Path("./varying_noise")
    df_path = log_path / "df"
    df_path.mkdir(exist_ok=True, parents=True)
    random_state = cfg.setting.random_state

    elapsed_prev = 0.0
    result_df_list = []
    for noise in cfg.setting.noise_list:
        estimated_policy_value_list = []
        ## define a dataset class
        dataset = SyntheticBanditDataset(
            n_actions=cfg.setting.n_actions,
            dim_context=cfg.setting.dim_context,
            beta=cfg.setting.beta,
            reward_type=cfg.setting.reward_type,
            n_cat_per_dim=cfg.setting.n_cat_per_dim,
            latent_param_mat_dim=cfg.setting.latent_param_mat_dim,
            n_cat_dim=cfg.setting.n_cat_dim,
            n_deficient_actions=int(cfg.setting.n_actions * cfg.setting.n_def_actions),
            reward_function=linear_reward_function,
            reward_std=noise,
            random_state=random_state,
        )

        ### approximate the ground-truth policy value using test bandit data
        test_bandit_data = dataset.obtain_batch_bandit_feedback(
            n_rounds=cfg.setting.n_test_data,
            n_users=cfg.setting.n_test_users,
        )
        policy_value = dataset.calc_ground_truth_policy_value(
            expected_reward=test_bandit_data["expected_reward"],
            action_dist=gen_eps_greedy(
                expected_reward=test_bandit_data["expected_reward"],
                eps=cfg.setting.eps,
            ),
        )
        for _ in tqdm(range(cfg.setting.n_seeds)):
            ## generate bandit data to perform OPE
            bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=cfg.setting.n_val_data, n_clusters=cfg.setting.n_clusters
            )
            pi_e = gen_eps_greedy(
                expected_reward=bandit_data["expected_reward"],
                eps=cfg.setting.eps,
            )

            ## obtain regression models
            f_x_a, q_x_a = train_reward_model_via_two_stage(
                bandit_data,
                bandit_data["clusters"],
                random_state=random_state + _,
            )

            ## OPE using validation data
            estimated_policy_values = run_ope(
                bandit_data=bandit_data,
                pi_e=pi_e,
                action_clusters=bandit_data["clusters"],
                f_x_a=f_x_a,
                q_x_a=q_x_a,
            )
            estimated_policy_value_list.append(estimated_policy_values)

        ## summarize results
        result_df = (
            DataFrame(DataFrame(estimated_policy_value_list).stack())
            .reset_index(1)
            .rename(columns={"level_1": "est", 0: "value"})
        )
        result_df["noise"] = noise
        result_df["se"] = (result_df.value - policy_value) ** 2
        result_df["bias"] = 0
        result_df["variance"] = 0
        result_df["true_value"] = policy_value
        sample_mean = DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
        for est_ in sample_mean["est"]:
            estimates = result_df.loc[result_df["est"] == est_, "value"].values
            mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
            mean_estimates = np.ones_like(estimates) * mean_estimates
            result_df.loc[result_df["est"] == est_, "bias"] = (
                policy_value - mean_estimates
            ) ** 2
            result_df.loc[result_df["est"] == est_, "variance"] = (
                estimates - mean_estimates
            ) ** 2
        result_df_list.append(result_df)

        elapsed = np.round((time() - start_time) / 60, 2)
        diff = np.round(elapsed - elapsed_prev, 2)
        logger.info(f"noise={noise}: {elapsed}min (diff {diff}min)")
        elapsed_prev = elapsed

    # aggregate all results
    result_df = pd.concat(result_df_list).reset_index(level=0)
    result_df.to_csv(df_path / "result_df.csv")


if __name__ == "__main__":
    main()
