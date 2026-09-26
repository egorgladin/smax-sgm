# SMax-SGM experiments

Python code for the variable fractional-delay filter and CIFAR-100 polynomial-activation experiments in *Sampled-Max Subgradient Method for Convex Finite-Max Optimization*.

Install dependencies with `python -m pip install -r requirements.txt`.

- `experiments/vfd_farrow/`: the reported budget comparison is in `run_early_budget_experiment.py`; it uses the SOCP history produced by `run_virtual_age_comparison.py`.
- `experiments/polynomial_activation/`: run `extract_cifar100_features.py` to create the feature cache, then `run_anytime_experiment.py` for the budget comparison.

Run a script with `--help` for its inputs and output paths. Downloaded data and generated results are not included.
