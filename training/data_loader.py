"""
Dataset loading and preprocessing.

Primary source is the Kaggle Credit Card Fraud Detection dataset
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807
transactions with a 0.172% fraud rate.

If that file is absent, a synthetic dataset with the same schema and a
comparable class imbalance is generated so the pipeline is runnable
immediately. Synthetic mode is clearly flagged in the run metadata so
results are never mistaken for the real benchmark.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
TARGET_COLUMN = "Class"


def generate_synthetic_data(n_rows: int = 100_000, fraud_rate: float = 0.0040, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic dataset that mirrors the structure of the Kaggle data.

    Fraudulent rows are drawn from shifted distributions on a subset of the
    V-features, which is what makes the classification problem learnable
    while preserving severe class imbalance.
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    # Legitimate transactions: standard normal PCA features, log-normal amounts.
    legit = rng.standard_normal((n_legit, 28))
    legit_amount = rng.lognormal(mean=3.2, sigma=1.1, size=n_legit)

    # Fraudulent transactions: a subset of features shifted, with deliberate
    # class overlap so the problem stays realistically hard.
    #
    # Two design choices keep this honest:
    #   1. Modest effect sizes (~0.9 sigma) rather than cleanly separated blobs.
    #   2. A "camouflaged" subset of fraud drawn from the legitimate
    #      distribution — real fraudsters actively mimic normal behaviour, and
    #      this is what caps recall below 1.0 on any real dataset.
    fraud = rng.standard_normal((n_fraud, 28))
    discriminative = [2, 3, 9, 10, 11, 13, 15, 16, 17]  # V3, V4, V10, V11, ...
    for col in discriminative:
        shift = rng.normal(loc=2.2, scale=0.8, size=n_fraud)
        fraud[:, col] += shift

    # ~10% of fraud is indistinguishable from legitimate traffic.
    n_camouflaged = int(n_fraud * 0.10)
    if n_camouflaged:
        camo_idx = rng.choice(n_fraud, size=n_camouflaged, replace=False)
        fraud[camo_idx] = rng.standard_normal((n_camouflaged, 28))

    # Label noise: a small fraction of legitimate rows look anomalous, which
    # is what generates the false positives any real fraud model produces.
    n_noisy = int(n_legit * 0.0015)
    if n_noisy:
        noisy_idx = rng.choice(n_legit, size=n_noisy, replace=False)
        for col in discriminative[:5]:
            legit[noisy_idx, col] += rng.normal(loc=2.2, scale=0.7, size=n_noisy)

    fraud_amount = rng.lognormal(mean=3.6, sigma=1.3, size=n_fraud)

    features = np.vstack([legit, fraud])
    amounts = np.concatenate([legit_amount, fraud_amount])
    labels = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])
    times = rng.uniform(0, 172_800, size=n_rows)  # two days in seconds

    df = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = times
    df["Amount"] = np.round(amounts, 2)
    df[TARGET_COLUMN] = labels

    # Shuffle so the class ordering isn't an artefact of construction.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def load_dataset(data_path: Path, allow_synthetic: bool = True) -> tuple[pd.DataFrame, bool]:
    """
    Returns:
        (dataframe, is_synthetic)
    """
    if data_path.exists():
        logger.info("Loading real dataset from %s", data_path)
        df = pd.read_csv(data_path)
        missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
        return df[FEATURE_COLUMNS + [TARGET_COLUMN]], False

    if not allow_synthetic:
        raise FileNotFoundError(
            f"{data_path} not found. Download creditcard.csv from Kaggle into data/."
        )

    logger.warning(
        "%s not found — generating synthetic dataset. "
        "Download the Kaggle dataset for benchmark-comparable results.",
        data_path,
    )
    return generate_synthetic_data(), True


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.20,
    seed: int = 42,
) -> dict:
    """
    Stratified three-way split: train / validation / test.

    Stratification is essential here — with a fraud rate below 1%, a random
    split can easily produce a test set containing almost no positive cases.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    # val_size is expressed as a fraction of the original dataset.
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, stratify=y_temp, random_state=seed
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": FEATURE_COLUMNS,
    }


def compute_baseline_stats(X_train: pd.DataFrame, n_bins: int = 10) -> dict:
    """
    Capture the training feature distributions used as the drift baseline.

    For each feature we store decile bin edges and the proportion of training
    rows in each bin. At inference time PSI compares live traffic against these.
    """
    stats: dict = {"feature_names": list(X_train.columns), "features": {}}

    for col in X_train.columns:
        values = X_train[col].to_numpy(dtype=float)

        # Quantile-based edges adapt to each feature's own scale.
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 2:  # constant feature — skip
            continue
        edges[0] = -np.inf
        edges[-1] = np.inf

        counts, _ = np.histogram(values, bins=edges)
        percents = counts / max(counts.sum(), 1)

        stats["features"][col] = {
            "bin_edges": edges.tolist(),
            "bin_percents": percents.tolist(),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return stats
