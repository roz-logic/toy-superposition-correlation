# Author: Oz (Özge Tepeli)
# Project: Toy superposition with correlated sparse features
# Description: Tied-weight bottleneck model x_hat = ReLU(W^T W x + b) and
#              geometry metrics for evaluating learned feature columns.
# Design notes: Using tied weights W of shape (m, n) means column i of W is
#               the single direction in m-dimensional hidden space assigned to
#               input feature i.
# References: Toy models of superposition setup Elhage et al., "Toy Models of Superposition", Transformer Circuits Thread, 2022 (https://transformer-circuits.pub/2022/toy_model/index.html)

import math
import torch
import torch.nn as nn


# =============================================================
# PLAIN-ENGLISH EXPLANATION OF THE LOSS AND METRICS
# =============================================================
# 1. Reconstruction MSE Loss (reconstruction_mse_loss):
#    For every input vector x, the model compresses its n = 20 features into
#    m = 5 numbers (W x) and then reconstructs an estimate x_hat = ReLU(W^T W x + b).
#    Mean Squared Error (MSE) is the average of (x_hat_i - x_i)^2 across all
#    20 features and all rows in the batch. It penalises the model equally for
#    failing to reconstruct an active feature (under-predicting) and for spilling
#    activity onto an inactive feature (interference).
#
# 2. Within-Pair Cosine Similarity (within_pair_cosine):
#    Each feature i has a direction in the 5D bottleneck given by column i of W.
#    For each correlated pair (0 and 1, 2 and 3, ..., 18 and 19), we divide both
#    columns by their lengths (L2 norms) and take their dot product.
#    A value near +1 means the two paired features point in almost the same direction;
#    a value near 0 means they are nearly orthogonal; a value near -1 means they
#    point in opposite directions. We average this signed value across the 10 pairs.
#
# 3. Between-Pair Mean Absolute Cosine Similarity (between_pair_abs_cosine):
#    As a baseline comparison, we look at pairs of features that belong to DIFFERENT
#    pairs (for example, feature 0 and feature 2) and are statistically independent.
#    Because 20 features crammed into 5 dimensions cannot all be mutually orthogonal,
#    even independent features have some non-zero overlap. Taking the average of
#    |cos(W_i, W_j)| across all 180 out-of-pair combinations tells us the background
#    level of geometric overlap between unpaired features in the 5D bottleneck.
#
# 4. Number of Represented Features (count_represented_features):
#    The length (L2 norm) of column i of W measures how strongly feature i is
#    written into the hidden space. If the network ignores a feature, its column
#    norm stays near 0. In this experiment, we use a chosen threshold of 0.5 and
#    count a feature as represented if its column norm is strictly above 0.5.


# =============================================================
# PART 1: TIED-WEIGHT BOTTLENECK MODEL
# =============================================================

class TiedBottleneckModel(nn.Module):
    """
    Bottleneck autoencoder with tied weights:
      x_hat = ReLU(W^T W x + b)
    where W has shape (m_hidden, n_features) and b has shape (n_features,).
    """

    def __init__(self, n_features: int = 20, m_hidden: int = 5) -> None:
        super().__init__()
        self.n_features = n_features
        self.m_hidden = m_hidden

        # W has shape (m, n): each column W[:, i] is feature i's vector in hidden space
        self.W = nn.Parameter(torch.empty(m_hidden, n_features))
        self.b = nn.Parameter(torch.zeros(n_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Scale initial weights by 1 / sqrt(n_features) so initial activations are small and stable
        nn.init.normal_(self.W, mean=0.0, std=1.0 / math.sqrt(self.n_features))
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For a batch x of row vectors with shape (batch_size, n):
        # h = x @ W^T computes W x for each row, giving shape (batch_size, m)
        # h @ W computes W^T (W x) for each row, giving shape (batch_size, n)
        h = x @ self.W.t()
        x_hat = torch.relu(h @ self.W + self.b)
        return x_hat


# =============================================================
# PART 2: LOSS AND GEOMETRY METRIC FUNCTIONS
# =============================================================

def reconstruction_mse_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error across all batch elements and all n features.
    """
    assert x_hat.shape == x.shape, f"shape mismatch: {x_hat.shape} vs {x.shape}"
    return torch.mean((x_hat - x) ** 2)


def normalise_columns(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Divides each column of W by its L2 norm, clamping the denominator at eps
    so a dead (zero) column never produces NaN values.
    """
    assert W.ndim == 2, f"expected 2D weight matrix W, got shape {W.shape}"
    col_norms = torch.linalg.vector_norm(W, ord=2, dim=0, keepdim=True)
    return W / torch.clamp(col_norms, min=eps)


def within_pair_cosine(W: torch.Tensor) -> float:
    """
    Computes the signed cosine similarity between paired columns (2k, 2k+1) of W,
    averaged across all n // 2 pairs.
    """
    n_features = W.shape[1]
    assert n_features % 2 == 0, "number of columns in W must be even"
    unit_cols = normalise_columns(W.detach())

    even_cols = unit_cols[:, 0::2]
    odd_cols = unit_cols[:, 1::2]
    # Column-wise dot product between each even column and its odd partner
    pair_cosines = torch.sum(even_cols * odd_cols, dim=0)
    return float(pair_cosines.mean().item())


def between_pair_abs_cosine(W: torch.Tensor) -> float:
    """
    Computes the mean absolute cosine similarity |cos(W[:, i], W[:, j])| across all
    feature pairs (i, j) with i < j that belong to different input pairs.
    """
    n_features = W.shape[1]
    assert n_features >= 4 and n_features % 2 == 0, "need at least 2 pairs (4 features)"
    unit_cols = normalise_columns(W.detach())
    cos_matrix = torch.abs(unit_cols.t() @ unit_cols)

    out_of_pair_values = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if (i // 2) != (j // 2):
                out_of_pair_values.append(cos_matrix[i, j].item())

    assert len(out_of_pair_values) > 0, "no out-of-pair feature combinations found"
    return float(sum(out_of_pair_values) / len(out_of_pair_values))


def count_represented_features(W: torch.Tensor, threshold: float = 0.5) -> int:
    """
    Counts how many columns of W have an L2 norm strictly greater than threshold.
    """
    col_norms = torch.linalg.vector_norm(W.detach(), ord=2, dim=0)
    return int(torch.sum(col_norms > threshold).item())


# =============================================================
# PART 3: HAND-CHECKED TESTS FOR MODEL AND METRICS
# =============================================================

def run_model_tests() -> None:
    print("Running model.py sanity checks...")

    # Test 1: Parameter shapes and hand-calculated forward pass
    model = TiedBottleneckModel(n_features=20, m_hidden=5)
    assert model.W.shape == (5, 20), f"expected W shape (5, 20), got {model.W.shape}"
    assert model.b.shape == (20,), f"expected b shape (20,), got {model.b.shape}"

    tiny_model = TiedBottleneckModel(n_features=2, m_hidden=1)
    with torch.no_grad():
        tiny_model.W.copy_(torch.tensor([[2.0, -1.0]]))
        tiny_model.b.copy_(torch.tensor([0.5, 0.5]))
    # W^T W = [[4, -2], [-2, 1]]. For x = [1, 0], W^T W x + b = [4.5, -1.5].
    # After ReLU, expected output is [4.5, 0.0].
    x_test = torch.tensor([[1.0, 0.0]])
    x_hat_test = tiny_model(x_test)
    expected_out = torch.tensor([[4.5, 0.0]])
    assert torch.allclose(x_hat_test, expected_out), (
        f"hand-checked forward pass failed: got {x_hat_test}, expected {expected_out}"
    )
    print("\u2713 Test 1 passed: W shape (5, 20) and hand-checked ReLU(W^T W x + b) forward pass verified")

    # Test 2: Hand-checked MSE loss
    pred = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.0, 0.0]])
    # Squared errors are 0^2 = 0 and 2^2 = 4; mean over 2 elements is 2.0
    mse_val = reconstruction_mse_loss(pred, target).item()
    assert abs(mse_val - 2.0) < 1e-6, f"expected MSE of 2.0, got {mse_val}"
    print("\u2713 Test 2 passed: reconstruction_mse_loss matches hand calculation")

    # Test 3: Hand-checked within_pair_cosine and between_pair_abs_cosine
    # 4 features (2 pairs) in 2D space:
    # Pair 0: col 0 = [1, 0], col 1 = [2, 0] -> parallel, cosine = +1.0
    # Pair 1: col 2 = [0, 1], col 3 = [0, -1] -> antiparallel, cosine = -1.0
    # Mean signed within-pair cosine = (1.0 + (-1.0)) / 2 = 0.0
    # Between-pair combinations compare [1, 0] or [2, 0] against [0, 1] or [0, -1]:
    # all 4 cross-pair dot products are 0.0 -> mean absolute between-pair cosine = 0.0
    W_toy = torch.tensor([
        [1.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0],
    ])
    w_cos = within_pair_cosine(W_toy)
    b_cos = between_pair_abs_cosine(W_toy)
    assert abs(w_cos - 0.0) < 1e-6, f"expected within-pair cosine 0.0, got {w_cos}"
    assert abs(b_cos - 0.0) < 1e-6, f"expected between-pair abs cosine 0.0, got {b_cos}"
    print("\u2713 Test 3 passed: within_pair_cosine and between_pair_abs_cosine match hand calculation")

    # Test 4: Hand-checked count_represented_features
    # Column 0 has norm sqrt(0.3^2 + 0.4^2) = 0.5 (not strictly above 0.5)
    # Column 1 has norm sqrt(0.3^2 + 0.5^2) = 0.5831 (strictly above 0.5)
    # Column 2 has norm 0.1 (below 0.5)
    # Column 3 has norm 1.0 (strictly above 0.5)
    W_norms = torch.tensor([
        [0.3, 0.3, 0.1, 1.0],
        [0.4, 0.5, 0.0, 0.0],
    ])
    rep_count = count_represented_features(W_norms, threshold=0.5)
    assert rep_count == 2, f"expected 2 represented features with norm > 0.5, got {rep_count}"
    print("\u2713 Test 4 passed: count_represented_features strictly applies the 0.5 column norm threshold")


if __name__ == "__main__":
    run_model_tests()
