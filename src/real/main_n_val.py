from logging import getLogger
from pathlib import Path
from time import time
import warnings

from dataset import ExtremeBanditDataset
import hydra
import numpy as np
from omegaconf import DictConfig
from ope import run_ope
from ope import train_clustering
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
    log_path = Path("./varying_n_val_data")
    df_path = log_path / "df"
    df_path.mkdir(exist_ok=True, parents=True)
    random_state = cfg.setting.random_state

    elapsed_prev = 0.0
    result_df_list = []
    for n_val_data in cfg.setting.n_val_data_list:
        estimated_policy_value_list = []
        ## define a dataset class
        dataset = ExtremeBanditDataset(
            dataset_name=cfg.setting.dataset,
            max_reward_noise=cfg.setting.max_reward_noise,
        )

        for _ in tqdm(range(cfg.setting.n_seeds)):
            # split the original data into training and evaluation sets
            bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=n_val_data,
                beta=cfg.setting.beta,
            )
            pi_e = gen_eps_greedy(
                expected_reward=bandit_data["expected_reward"],
                eps=cfg.setting.eps,
            )
            policy_value = dataset.calc_ground_truth_policy_value(
                action_dist=pi_e,
                expected_reward=bandit_data["expected_reward"],
            )

            ## perform action clustering
            action_clusters = train_clustering(
                bandit_data=bandit_data,
                n_clusters=cfg.setting.n_clusters,
                random_state=12345 + _,
            )

            f_x_a, q_x_a = train_reward_model_via_two_stage(
                bandit_data,
                action_clusters,
                random_state=random_state + _,
            )

            ## OPE using validation data
            estimated_policy_values = run_ope(
                bandit_data=bandit_data,
                action_clusters=action_clusters,
                pi_e=pi_e,
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
        result_df["n_val_data"] = n_val_data
        result_df["se"] = (result_df.value - policy_value) ** 2
        result_df["policy_value"] = policy_value
        result_df["bias"] = 0
        result_df["variance"] = 0
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
        logger.info(f"n_val_data={n_val_data}: {elapsed}min (diff {diff}min)")
        elapsed_prev = elapsed

    # aggregate all results
    result_df = pd.concat(result_df_list).reset_index(level=0)
    result_df.to_csv(df_path / "result_df.csv")


if __name__ == "__main__":
    main()
