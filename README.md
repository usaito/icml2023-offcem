# Off-Policy Evaluation for Large Action Spaces via Conjunct Effect Modeling

This repository contains the code used for the experiments in ["Off-Policy Evaluation for Large Action Spaces via Conjunct Effect Modeling"](https://arxiv.org/abs/2305.08062) by [Yuta Saito](https://usait0.com/en/), Qingyang Ren, and [Thorsten Joachims](https://www.cs.cornell.edu/people/tj/).

Note that our experiment implementation is based on the [Open Bandit Pipeline](https://github.com/st-tech/zr-obp), a modular Python package for off-policy evaluation.

## Abstract
We study off-policy evaluation (OPE) of contextual bandit policies for large discrete action spaces where conventional importance-weighting approaches suffer from excessive variance. To circumvent this variance issue, we propose a new estimator, called *OffCEM*, that is based on the *conjunct effect model* (CEM), a novel decomposition of the causal effect into a cluster effect and a residual effect. OffCEM applies importance weighting only to action clusters and addresses the residual causal effect through model-based reward estimation. We show that the proposed estimator is unbiased under a new assumption, called *local correctness*, which only requires that the residual-effect model preserves the relative expected reward differences of the actions within each cluster. To best leverage the CEM and local correctness, we also propose a new two-step procedure for performing model-based estimation that minimizes bias in the first step and variance in the second step. We find that the resulting OffCEM estimator substantially improves bias and variance compared to a range of conventional estimators. Experiments demonstrate that OffCEM provides substantial improvements in OPE especially in the presence of many actions.

## Citation

```
@article{saito2023off,
  title={Off-Policy Evaluation for Large Action Spaces via Conjunct Effect Modeling},
  author={Saito, Yuta and Ren, Qingyang and Joachims, Thorsten},
  journal={arXiv preprint arXiv:2305.08062},
  year={2023}
}
```

## Requirements and Setup

The Python environment is built using [poetry](https://github.com/python-poetry/poetry). You can build the same environment as in our experiments by cloning the repository and running `poetry install` directly under the folder (if you have not installed `poetry` yet, please run `pip install poetry` first).

```bash
# clone the repository
git clone https://github.com/usaito/icml2023-offcem.git
cd src

# install poetry
pip install poetry

# build the environment with poetry
poetry install
```

The versions of Python and necessary packages are specified as follows (from [pyproject.toml](./pyproject.toml)).

```
[tool.poetry.dependencies]
python = ">=3.9,<3.10"
obp = "^0.5.5"
pandas = "^1.4.3"
numpy = "1.22.4"
scikit-learn = "1.0.2"
scipy = "1.7.3"
matplotlib = "^3.5.2"
seaborn = "^0.11.2"
hydra-core = "1.0.7"
```

## Running the Code

The experimental workflow is implemented using [Hydra](https://github.com/facebookresearch/hydra). The commands needed to reproduce the experiments are summarized below. Please move under the `src` directly first and then run the commands.

### Section 4.1: Synthetic Data

```bash
cd src

# How does OffCEM perform with varying sample sizes?
poetry run python synthetic/main_n_val.py

# How does OffCEM perform with varying numbers of actions?
poetry run python synthetic/main_n_actions.py

# How does OffCEM perform with varying numbers of deficient actions?
poetry run python synthetic/main_n_def_actions.py

# How does OffCEM perform with varying noise levels? (in the Appendix)
poetry run python synthetic/main_noise.py

# How does OffCEM perform with varying evaluation policies? (in the Appendix)
poetry run python synthetic/main_eps.py

# How does OffCEM perform with varying numbers of clusters? (in the Appendix)
poetry run python synthetic/main_n_clusters.py
```

### Section 4.2: Real-World Classification Data

We use "Wiki10-31K" and "EURLex-4K" from [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html). Please download the above datasets from the repository and put them under `./src/real/` as follows.

```
src/
　├ synthetic/
　├ real/
　 　└ data/
　 　   ├── eurlex/
            ├ train.txt
            ├ test/text
　 　   └── wiki/
            ├ train.txt
            ├ test/text
```

Then, run the following.

```bash
cd src

poetry run python real/main_n_val.py setting.dataset=eurlex,wiki -m
```
