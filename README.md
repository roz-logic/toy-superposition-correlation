
# Correlated Inputs in a Toy Bottleneck Network

Author: Oz (Özge Tepeli)

## Question

When twenty sparse features are arranged into ten correlated pairs, at what level of correlation does a bottleneck network begin to treat the two features in each pair as a single direction? The aim is to identify the point at which the network merges their weight directions.

## Experimental Design

Each input comprised twenty sparse features, grouped into ten pairs: (0, 1), (2, 3), and so forth. To avoid confounding correlation with feature frequency, the marginal firing rate for each feature was held constant at 0.05, with active values drawn uniformly from the interval [0, 1]. Within each pair, feature A was activated independently at the given rate. Feature B either replicated the activity of feature A with probability ρ, or was sampled independently from the same distribution with probability 1 - ρ. Empirical validation via unit tests (see data.py) confirmed that both the firing rates and within-pair correlations closely matched the intended values over 200,000 samples.

The model (model.py) was a bottleneck network with tied weights, defined as \(\hat{x} = \mathrm{ReLU}(W^T W x + b)\), where \(W\) had dimensions 5 × 20 and the bias vector \(b\) had length 20. Training minimised mean squared reconstruction error. Six values of ρ (0.0, 0.2, 0.4, 0.6, 0.8, 0.95) were examined, with ten random seeds per condition (101–110), leading to a total of sixty trained models. Each model was trained for 5,000 steps with a batch size of 1,024, using the Adam optimiser (learning rate 0.001). For each weight matrix, the mean signed cosine similarity between columns in a pair (within_pair_cos), the mean absolute cosine similarity between columns from different pairs (between_pair_abs_cos), and the number of features with column L2 norm exceeding 0.5 (n_represented) were recorded.

## Results

![Bottleneck feature geometry across pairwise correlation rho](results/superposition_correlation_figure.png)

| `rho` | Seeds | Within-pair `cos` (mean +/- SD) | Between-pair `\|cos\|` (mean +/- SD) | Represented features `\|\|W_i\|\|_2 > 0.5` (mean +/- SD) | Final MSE (`x 10^-3`, mean +/- SD) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.00` | 10 | `-0.081 +/- 0.063` | `0.366 +/- 0.002` | `19.9 +/- 0.32` | `4.59 +/- 0.48` |
| `0.20` | 10 | `+0.376 +/- 0.037` | `0.353 +/- 0.008` | `19.9 +/- 0.32` | `4.66 +/- 0.28` |
| `0.40` | 10 | `+0.431 +/- 0.013` | `0.337 +/- 0.002` | `20.0 +/- 0.00` | `4.06 +/- 0.20` |
| `0.60` | 10 | `+0.578 +/- 0.074` | `0.318 +/- 0.009` | `20.0 +/- 0.00` | `3.56 +/- 0.32` |
| `0.80` | 10 | `+0.9999 +/- 0.0000` | `0.115 +/- 0.000` | `20.0 +/- 0.00` | `2.00 +/- 0.18` |
| `0.95` | 10 | `+0.980 +/- 0.063` | `0.113 +/- 0.001` | `19.9 +/- 0.32` | `0.94 +/- 0.33` |

For uncorrelated features (ρ = 0.0), the network represented nearly all twenty features within the five-dimensional bottleneck (mean 19.9 with column norm above 0.5). The mean within-pair cosine was -0.081. As ρ increased from 0.2 to 0.6, within-pair cosine rose from 0.376 to 0.578. At ρ = 0.6, most pairs displayed partial alignment (cosine between approximately 0.45 and 0.60), whilst in some seeds, certain pairs had already fully aligned (cosine of 1.00).

By ρ = 0.8, all pairs had merged completely, with mean within-pair cosine +0.9999 across all seeds. As only six ρ values were considered, the precise transition occurs between 0.6 and 0.8. Following merging, all twenty features retained column norms above 0.5 (at ρ = 0.8, each approximately 0.71, i.e., squared norm 0.50 or 1/√2). This ensures that simultaneous activation of both features in a pair produces unit length, avoiding artificial inflation of the reconstructed output.

Overlap between pairs (|cos|) declined from 0.366 at ρ = 0.0 to 0.115 at ρ = 0.8. At this latter value, the merged pair directions (as observed in seed 101) formed five orthogonal axes within the bottleneck, with each axis occupied by two features pointing in opposite directions (cosine = -1), and zero elsewhere. Thus, each pair had |cos| = 1 with a single other pair, and zero with the remaining eight (mean 1/9 ≈ 0.111).

At ρ = 0.95, nine of the ten seeds maintained within_pair_cos = 1.0000; in one instance (seed 106), a single pair failed to merge, resulting in mean cosine 0.800, only nineteen features represented above threshold, and a slightly higher mean squared error (0.00171 compared to 0.00086 in the remaining seeds). Inspection showed that in this case, one feature in the affected pair took a norm of 1.03, while its partner collapsed to 0.064 in the opposite direction (cosine = -0.999). This likely reflects a local minimum encountered during optimisation, and highlights the value of using multiple random seeds.

## Limitations

This experiment is illustrative and deliberately constrained, employing twenty features and five hidden units, synthetic uniformly sparse data, a single network architecture, and a fixed sparsity level (p = 0.05). The parameter grid is too coarse to determine precisely at which value of ρ the transition occurs between 0.6 and 0.8. No claims are made regarding the generalisability of these findings to larger or biological networks.

## Repository Structure and Reproducibility

The experiment is governed by four principal Python scripts: data.py, model.py, plot.py, and run_experiment.py.

```text
toy-superposition-correlation/
├── data.py
├── model.py
├── plot.py
├── run_experiment.py
├── requirements.txt
├── README.md
└── results/
    ├── experiment_results.csv
    ├── summary_by_rho.csv
    └── superposition_correlation_figure.png
```

To install the dependencies, run the unit tests, retrain all sixty models (which requires approximately eight and a half minutes on a typical laptop CPU), and regenerate the results from scratch, use:

```bash
pip install -r requirements.txt
python run_experiment.py
```

## Background and Reference

This methodology is based on the tied-weight ReLU toy model, and in particular the discussion of correlated and anticorrelated features, as well as the collapse of correlated features, in:

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., & Olah, C. (2022). *Toy Models of Superposition*. Transformer Circuits Thread. [https://transformer-circuits.pub/2022/toy_model/index.html](https://transformer-circuits.pub/2022/toy_model/index.html)

In their analysis, the authors show that, under tight bottleneck constraints, pairs of correlated features may collapse onto a shared principal component with weights of length close to 1/√2, rather than remaining in orthogonal directions. Here, this approach is extended by varying ρ systematically from 0.0 to 0.95, while holding marginal sparsity fixed at p = 0.05 across ten random seeds per condition.
