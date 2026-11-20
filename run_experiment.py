# Author: Oz (Özge Tepeli)
# Project: Toy superposition with correlated sparse features
# Description: Trains the m = 5, n = 20 tied bottleneck network across
#              rho in {0, 0.2, 0.4, 0.6, 0.8, 0.95} with 10 seeds per rho,
#              logs all metrics to CSV, and calls plot.py.
# Design notes: Runs all data and model sanity tests before starting training.
#               Uses a dedicated PyTorch Generator seeded per (rho, seed) run
#               so every run is reproducible and logged in the output CSV.
# References: Toy models of superposition setup Elhage et al., "Toy Models of Superposition", Transformer Circuits Thread, 2022 (https://transformer-circuits.pub/2022/toy_model/index.html)

import os
import time
import random
import numpy as np
import pandas as pd
import torch

from data import sample_correlated_batch, run_data_tests
from model import (
    TiedBottleneckModel,
    reconstruction_mse_loss,
    within_pair_cosine,
    between_pair_abs_cosine,
    count_represented_features,
    run_model_tests,
)
from plot import make_plots


# =============================================================
# PART 1: EXPERIMENT CONFIGURATION AND SEED CONTROL
# =============================================================

N_FEATURES = 20
M_HIDDEN = 5
P_ACTIVE = 0.05
RHOS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

STEPS = 5000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
NORM_THRESHOLD = 0.5

OUTDIR = "results"
RESULTS_CSV_PATH = os.path.join(OUTDIR, "experiment_results.csv")
SUMMARY_CSV_PATH = os.path.join(OUTDIR, "summary_by_rho.csv")
FIGURE_PATH = os.path.join(OUTDIR, "superposition_correlation_figure.png")


def set_all_seeds(seed: int) -> None:
    """
    Sets Python, NumPy, and PyTorch CPU seeds so weight initialisation and
    optimiser state trajectories are reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_seed_reproducibility() -> None:
    set_all_seeds(777)
    m1 = TiedBottleneckModel(n_features=N_FEATURES, m_hidden=M_HIDDEN)
    set_all_seeds(777)
    m2 = TiedBottleneckModel(n_features=N_FEATURES, m_hidden=M_HIDDEN)
    assert torch.allclose(m1.W, m2.W), "identical seeds did not produce identical initial weights"
    print("\u2713 Seed reproducibility check passed: identical seeds yield identical model weights")


# =============================================================
# PART 2: SINGLE TRAINING RUN
# =============================================================

def train_single_model(
    rho: float,
    seed: int,
    n_features: int = N_FEATURES,
    m_hidden: int = M_HIDDEN,
    p: float = P_ACTIVE,
    steps: int = STEPS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> dict:
    """
    Trains one TiedBottleneckModel for a single (rho, seed) combination and
    returns a dictionary of logged configuration values, losses, and metrics.
    """
    set_all_seeds(seed)
    # Offset the data generator seed by rho so batch draws are distinct across rho levels
    data_seed = seed * 10_000 + int(round(rho * 1_000))
    gen = torch.Generator().manual_seed(data_seed)

    model = TiedBottleneckModel(n_features=n_features, m_hidden=m_hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    initial_loss = None
    final_loss = None

    for step in range(steps):
        x = sample_correlated_batch(
            batch_size=batch_size,
            n_features=n_features,
            p=p,
            rho=rho,
            generator=gen,
        )
        x_hat = model(x)
        loss = reconstruction_mse_loss(x_hat, x)

        if step == 0:
            initial_loss = float(loss.item())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step == steps - 1:
            final_loss = float(loss.item())

    assert initial_loss is not None and final_loss is not None, "training loop did not record losses"
    assert np.isfinite(final_loss), f"non-finite final loss encountered at rho={rho}, seed={seed}"

    # Evaluate learned weight geometry on a detached copy of W
    W_learned = model.W.detach()
    col_norms = torch.linalg.vector_norm(W_learned, ord=2, dim=0)

    return {
        "rho": rho,
        "seed": seed,
        "data_seed": data_seed,
        "n_features": n_features,
        "m_hidden": m_hidden,
        "p_active": p,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "initial_mse_loss": round(initial_loss, 6),
        "final_mse_loss": round(final_loss, 6),
        "within_pair_cos": within_pair_cosine(W_learned),
        "between_pair_abs_cos": between_pair_abs_cosine(W_learned),
        "n_represented": count_represented_features(W_learned, threshold=NORM_THRESHOLD),
        "mean_col_norm": float(col_norms.mean().item()),
        "min_col_norm": float(col_norms.min().item()),
        "max_col_norm": float(col_norms.max().item()),
    }


def test_short_training_decreases_loss() -> None:
    """
    Quick sanity check that 150 steps of Adam actually reduce MSE loss on our data
    before launching the 60-run grid.
    """
    res = train_single_model(rho=0.0, seed=999, steps=150, batch_size=512)
    assert res["final_mse_loss"] < res["initial_mse_loss"], (
        f"loss failed to decrease in short sanity test: "
        f"{res['initial_mse_loss']} -> {res['final_mse_loss']}"
    )
    print("\u2713 Short training check passed: Adam updates decrease reconstruction MSE loss")


# =============================================================
# PART 3: FULL EXPERIMENT GRID (6 RHO LEVELS x 10 SEEDS = 60 RUNS)
# =============================================================

def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    # Run all invariant tests before spending CPU time on the experiment grid
    run_data_tests()
    run_model_tests()
    test_seed_reproducibility()
    test_short_training_decreases_loss()

    print("\nStarting full experiment grid:")
    print(f"  n_features = {N_FEATURES}, m_hidden = {M_HIDDEN}, p = {P_ACTIVE}")
    print(f"  rhos = {RHOS}")
    print(f"  seeds ({len(SEEDS)}) = {SEEDS}")
    print(f"  steps = {STEPS}, batch_size = {BATCH_SIZE}, lr = {LEARNING_RATE}\n")

    start_time = time.time()
    all_rows = []
    total_runs = len(RHOS) * len(SEEDS)
    run_counter = 0

    for rho in RHOS:
        rho_start = time.time()
        for seed in SEEDS:
            run_counter += 1
            row = train_single_model(rho=rho, seed=seed)
            all_rows.append(row)
        rho_elapsed = time.time() - rho_start
        print(
            f"Completed rho = {rho:.2f} ({len(SEEDS)} seeds) "
            f"[{run_counter}/{total_runs} runs] in {rho_elapsed:.1f}s"
        )

    total_elapsed = time.time() - start_time
    df_results = pd.DataFrame(all_rows)
    assert len(df_results) == total_runs, f"expected {total_runs} rows, got {len(df_results)}"
    assert df_results.isna().sum().sum() == 0, "unexpected missing values in experiment results"

    df_results.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\n\u2713 Saved all {len(df_results)} individual run records to {RESULTS_CSV_PATH}")

    summary_df = make_plots(
        results_csv_path=RESULTS_CSV_PATH,
        summary_csv_path=SUMMARY_CSV_PATH,
        figure_path=FIGURE_PATH,
    )

    print("\nSummary across 10 seeds per rho (mean and sample SD):")
    print(summary_df.to_string(index=False))
    print(f"\nTotal elapsed time: {total_elapsed:.2f} seconds ({total_elapsed / 60.0:.2f} minutes)")


if __name__ == "__main__":
    main()
