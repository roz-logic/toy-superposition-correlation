# Author: Oz (Özge Tepeli)
# Project: Toy superposition with correlated sparse features
# Description: Synthetic data generator for paired sparse features where marginal
#              activation rates stay fixed at p = 0.05 across all pairwise
#              copying probabilities rho.
# Design notes: If changing correlation also changes how often a feature fires,
#               feature frequency becomes a confound. Here feature B copies
#               feature A with probability rho and is drawn independently with
#               probability (1 - rho). Because both branches have the same
#               marginal distribution, the marginal rate p is invariant to rho.
# References: Toy models of superposition setup Elhage et al., "Toy Models of Superposition", Transformer Circuits Thread, 2022 (https://transformer-circuits.pub/2022/toy_model/index.html)

import torch


# =============================================================
# PART 1: SYNTHETIC DATA GENERATOR FOR PAIRED SPARSE FEATURES
# =============================================================

def sample_correlated_batch(
    batch_size: int,
    n_features: int = 20,
    p: float = 0.05,
    rho: float = 0.0,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Generates a batch of shape (batch_size, n_features) with paired features:
    (0, 1), (2, 3), ..., (n_features - 2, n_features - 1).
    Each feature is active with marginal probability p and takes a Uniform(0, 1)
    value when active. Within each pair, feature B copies feature A with
    probability rho, and is drawn independently otherwise.
    """
    assert n_features > 0 and n_features % 2 == 0, (
        f"n_features must be a positive even integer so features form pairs, got {n_features}"
    )
    assert 0.0 <= p <= 1.0, f"marginal probability p must be in [0, 1], got {p}"
    assert 0.0 <= rho <= 1.0, f"copying probability rho must be in [0, 1], got {rho}"

    n_pairs = n_features // 2

    # Draw feature A for every pair: active with probability p, magnitude Uniform(0, 1)
    active_a = (torch.rand((batch_size, n_pairs), generator=generator) < p).float()
    values_a = torch.rand((batch_size, n_pairs), generator=generator)
    feat_a = active_a * values_a

    # Draw an independent candidate for feature B from the exact same distribution
    active_b_indep = (torch.rand((batch_size, n_pairs), generator=generator) < p).float()
    values_b_indep = torch.rand((batch_size, n_pairs), generator=generator)
    feat_b_indep = active_b_indep * values_b_indep

    # With probability rho, feature B copies feature A's activity; otherwise it uses the independent draw
    copy_switch = (torch.rand((batch_size, n_pairs), generator=generator) < rho).float()
    feat_b = copy_switch * feat_a + (1.0 - copy_switch) * feat_b_indep

    # Interleave columns so pair k sits at indices (2*k, 2*k + 1)
    batch = torch.stack([feat_a, feat_b], dim=2).reshape(batch_size, n_features)
    return batch


# =============================================================
# PART 2: DIAGNOSTIC HELPERS FOR MARGINAL RATE AND CORRELATION
# =============================================================

def empirical_marginal_rates(batch: torch.Tensor) -> torch.Tensor:
    """
    Returns the fraction of rows in which each feature is strictly positive.
    Used to verify that changing rho does not shift individual feature firing rates.
    """
    assert batch.ndim == 2, f"expected 2D batch tensor, got shape {batch.shape}"
    active_indicator = (batch > 0.0).float()
    return active_indicator.mean(dim=0)


def empirical_corr_matrix(batch: torch.Tensor) -> torch.Tensor:
    """
    Computes the sample Pearson correlation matrix across the columns of batch.
    Used to check that within-pair correlation matches rho and between-pair
    correlation stays near zero.
    """
    assert batch.ndim == 2 and batch.shape[0] > 1, "need at least 2 rows to compute correlation"
    centred = batch - batch.mean(dim=0, keepdim=True)
    cov = (centred.t() @ centred) / float(batch.shape[0] - 1)
    std = batch.std(dim=0, unbiased=True)
    assert torch.all(std > 0.0), "at least one feature had zero variance in the sample, stop"
    corr = cov / (std[:, None] * std[None, :])
    return corr


# =============================================================
# PART 3: HAND-CHECKED TESTS FOR DATA INVARIANTS
# =============================================================

def run_data_tests() -> None:
    print("Running data.py sanity checks...")

    # Test 1: Shape, value bounds, and exact copying when rho = 1.0
    gen = torch.Generator().manual_seed(2026)
    b_copy = sample_correlated_batch(batch_size=500, n_features=20, p=0.05, rho=1.0, generator=gen)
    assert b_copy.shape == (500, 20), f"expected shape (500, 20), got {b_copy.shape}"
    assert torch.all(b_copy >= 0.0) and torch.all(b_copy <= 1.0), "feature values fell outside [0, 1]"
    assert torch.allclose(b_copy[:, 0::2], b_copy[:, 1::2]), (
        "at rho=1.0, odd columns must exactly equal even partner columns"
    )
    print("\u2713 Test 1 passed: shape, [0, 1] range, and exact pair copying at rho = 1.0 verified")

    # Test 2: Marginal activation rate stays at p = 0.05 across rho values on a large sample
    n_large = 200_000
    target_p = 0.05
    for rho_test in [0.0, 0.4, 0.95]:
        gen_m = torch.Generator().manual_seed(100 + int(rho_test * 100))
        large_batch = sample_correlated_batch(
            batch_size=n_large, n_features=20, p=target_p, rho=rho_test, generator=gen_m
        )
        rates = empirical_marginal_rates(large_batch)
        max_drift = torch.max(torch.abs(rates - target_p)).item()
        assert max_drift < 0.003, (
            f"marginal firing rate drifted at rho={rho_test}: max drift {max_drift:.5f} exceeds 0.003"
        )
    print("\u2713 Test 2 passed: marginal activation probability stays at p = 0.05 across all tested rho values")

    # Test 3: Realised within-pair correlation matches rho and between-pair correlation is near zero
    for rho_test in [0.0, 0.2, 0.6, 0.95]:
        gen_c = torch.Generator().manual_seed(500 + int(rho_test * 100))
        large_batch = sample_correlated_batch(
            batch_size=n_large, n_features=20, p=target_p, rho=rho_test, generator=gen_c
        )
        corr = empirical_corr_matrix(large_batch)

        within_corrs = [corr[2 * k, 2 * k + 1].item() for k in range(10)]
        between_corrs = [
            corr[i, j].item()
            for i in range(20)
            for j in range(i + 1, 20)
            if (i // 2) != (j // 2)
        ]
        mean_within = sum(within_corrs) / len(within_corrs)
        mean_abs_between = sum(abs(c) for c in between_corrs) / len(between_corrs)

        assert abs(mean_within - rho_test) < 0.01, (
            f"realised within-pair correlation {mean_within:.4f} did not match target rho {rho_test}"
        )
        assert mean_abs_between < 0.01, (
            f"between-pair correlation {mean_abs_between:.4f} should be near zero"
        )
    print("\u2713 Test 3 passed: realised within-pair correlation matches rho and between-pair correlation is near 0")


if __name__ == "__main__":
    run_data_tests()
