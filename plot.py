# Author: Oz (Özge Tepeli)
# Project: Toy superposition with correlated sparse features
# Description: Summarises run-level CSV metrics across seeds (mean and sample SD)
#              and plots the two-panel figure.
# Design notes: Error bands show mean +/- 1 sample standard deviation (ddof=1)
#               across the 10 random seeds at each rho.
# References: Toy models of superposition setup Elhage et al., "Toy Models of Superposition", Transformer Circuits Thread, 2022 (https://transformer-circuits.pub/2022/toy_model/index.html)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================
# PART 1: AGGREGATE RUN-LEVEL RESULTS BY RHO
# =============================================================

def summarise_by_rho(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups run-level results by rho and computes the mean and sample standard
    deviation (ddof=1) across seeds for the geometry and representation metrics.
    """
    required_cols = {
        "rho",
        "seed",
        "within_pair_cos",
        "between_pair_abs_cos",
        "n_represented",
        "final_mse_loss",
    }
    missing = required_cols - set(df.columns)
    assert len(missing) == 0, f"missing expected columns in results DataFrame: {missing}"

    grouped = (
        df.groupby("rho", as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            within_cos_mean=("within_pair_cos", "mean"),
            within_cos_sd=("within_pair_cos", lambda s: float(np.std(s, ddof=1))),
            between_abs_cos_mean=("between_pair_abs_cos", "mean"),
            between_abs_cos_sd=("between_pair_abs_cos", lambda s: float(np.std(s, ddof=1))),
            n_rep_mean=("n_represented", "mean"),
            n_rep_sd=("n_represented", lambda s: float(np.std(s, ddof=1))),
            mse_mean=("final_mse_loss", "mean"),
            mse_sd=("final_mse_loss", lambda s: float(np.std(s, ddof=1))),
        )
        .sort_values("rho")
        .reset_index(drop=True)
    )
    return grouped


def test_summarise_by_rho() -> None:
    print("Running plot.py sanity checks...")
    toy_df = pd.DataFrame({
        "rho": [0.0, 0.0, 0.5, 0.5],
        "seed": [1, 2, 1, 2],
        "within_pair_cos": [0.1, 0.3, -0.2, -0.4],
        "between_pair_abs_cos": [0.2, 0.2, 0.3, 0.5],
        "n_represented": [10, 12, 14, 16],
        "final_mse_loss": [0.01, 0.01, 0.02, 0.02],
    })
    summary = summarise_by_rho(toy_df)
    assert len(summary) == 2, f"expected 2 summary rows, got {len(summary)}"
    assert abs(summary.loc[0, "within_cos_mean"] - 0.2) < 1e-6, "mean calculation failed"
    # Sample SD of [0.1, 0.3] with ddof=1 is sqrt(((0.1-0.2)^2 + (0.3-0.2)^2) / 1) = sqrt(0.02)
    expected_sd = float(np.sqrt(0.02))
    assert abs(summary.loc[0, "within_cos_sd"] - expected_sd) < 1e-6, "sample SD calculation failed"
    print("\u2713 summarise_by_rho check passed: mean and sample SD (ddof=1) match hand calculation")


# =============================================================
# PART 2: TWO-PANEL FIGURE GENERATION
# =============================================================

def make_plots(
    results_csv_path: str = "results/experiment_results.csv",
    summary_csv_path: str = "results/summary_by_rho.csv",
    figure_path: str = "results/superposition_correlation_figure.png",
) -> pd.DataFrame:
    """
    Loads the run-level CSV, computes seed summaries, writes a summary CSV,
    and saves the two-panel figure:
      Panel A: Mean within-pair cosine similarity (and between-pair mean |cos|
               comparison) against rho, with shaded bands showing mean +/- 1 SD.
      Panel B: Number of represented features (column norm > 0.5) against rho,
               with shaded band showing mean +/- 1 SD.
    """
    test_summarise_by_rho()

    assert os.path.exists(results_csv_path), (
        f"results CSV not found at {results_csv_path}, run run_experiment.py first"
    )
    df = pd.read_csv(results_csv_path)
    summary = summarise_by_rho(df)
    summary.to_csv(summary_csv_path, index=False)
    print(f"\u2713 Saved summary table by rho to {summary_csv_path}")

    rhos = summary["rho"].to_numpy()

    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )

    # Panel A: Cosine similarity vs rho (mean +/- 1 SD across 10 seeds)
    ax0 = axes[0]
    w_mean = summary["within_cos_mean"].to_numpy()
    w_sd = summary["within_cos_sd"].to_numpy()
    b_mean = summary["between_abs_cos_mean"].to_numpy()
    b_sd = summary["between_abs_cos_sd"].to_numpy()

    ax0.axhline(0.0, color="grey", linestyle="dotted", linewidth=1.0)
    ax0.plot(
        rhos,
        w_mean,
        marker="o",
        color="#8B2635",
        linewidth=2.0,
        label="Within-pair signed cos (mean +/- 1 SD)",
    )
    ax0.fill_between(
        rhos, w_mean - w_sd, w_mean + w_sd, color="#8B2635", alpha=0.20
    )

    ax0.plot(
        rhos,
        b_mean,
        marker="s",
        linestyle="dashed",
        color="#2E5A88",
        linewidth=1.8,
        label="Between-pair |cos| comparison (mean +/- 1 SD)",
    )
    ax0.fill_between(
        rhos, b_mean - b_sd, b_mean + b_sd, color="#2E5A88", alpha=0.18
    )

    ax0.set_title("A. Feature Alignment Across Pairwise Correlation")
    ax0.set_xlabel("Pairwise copying probability (rho)")
    ax0.set_ylabel("Cosine similarity between columns of W")
    ax0.set_xticks(rhos)
    ax0.set_ylim(-1.05, 1.05)
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="best", frameon=True, fontsize=9)

    # Panel B: Number of represented features (column norm > 0.5) vs rho
    ax1 = axes[1]
    rep_mean = summary["n_rep_mean"].to_numpy()
    rep_sd = summary["n_rep_sd"].to_numpy()

    ax1.axhline(5.0, color="grey", linestyle="dotted", linewidth=1.0, label="Bottleneck size (m = 5)")
    ax1.axhline(20.0, color="black", linestyle="dashed", linewidth=1.0, label="Total features (n = 20)")
    ax1.plot(
        rhos,
        rep_mean,
        marker="o",
        color="#E07B54",
        linewidth=2.0,
        label="Represented features (mean +/- 1 SD)",
    )
    ax1.fill_between(
        rhos, rep_mean - rep_sd, rep_mean + rep_sd, color="#E07B54", alpha=0.25
    )

    ax1.set_title("B. Represented Features (||W_i||_2 > 0.5)")
    ax1.set_xlabel("Pairwise copying probability (rho)")
    ax1.set_ylabel("Number of features with column norm > 0.5")
    ax1.set_xticks(rhos)
    ax1.set_ylim(0, 21.5)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=250)
    plt.close(fig)

    assert os.path.exists(figure_path) and os.path.getsize(figure_path) > 0, (
        f"failed to write non-empty figure to {figure_path}"
    )
    print(f"\u2713 Saved two-panel figure to {figure_path}")
    return summary


if __name__ == "__main__":
    make_plots()
