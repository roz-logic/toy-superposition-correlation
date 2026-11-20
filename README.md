# Correlated Inputs in a Toy Bottleneck Network

## Overview

This project investigates how correlations between input features affect the geometry of learned representations in a toy superposition model.

The experiment uses synthetic sparse feature data with controlled pairwise correlations and a tied-weight ReLU bottleneck network. By varying the correlation parameter, we examine how the network's learned weight geometry changes under different input correlation structures.

## Research Question

How do correlations between input features affect the geometry of learned representations in a toy bottleneck network?

## Experimental Design

The experiment varies the pairwise input correlation parameter across six conditions:

`rho = 0.0, 0.2, 0.4, 0.6, 0.8, 0.95`

For each condition, ten random seeds (101–110) are used, resulting in 60 trained models.

Each model is trained for 5,000 steps with a batch size of 1,024 using Adam with a learning rate of 0.001.

## Repository Structure

```text
toy-superposition-correlation/
│
├── data.py
├── model.py
├── plot.py
├── run_experiment.py
├── requirements.txt
├── README.md
│
└── results/
    ├── experiment_results.csv
    ├── summary_by_rho.csv
    └── superposition_correlation_figure.png
```

## Requirements

The experiment dependencies are listed in `requirements.txt`:

- PyTorch 2.2.2
- NumPy 1.26.4
- pandas 2.2.2
- Matplotlib 3.8.4

## Results

The repository includes:

- `experiment_results.csv`: individual experiment-run results.
- `summary_by_rho.csv`: aggregated results for each correlation condition.
- `superposition_correlation_figure.png`: visualization of the experiment results.

## Reproducibility

To reproduce the experiment, install the dependencies and run:

```bash
pip install -r requirements.txt
python run_experiment.py
```

The experiment runner generates the result files and summary figure.

## Background

This project is inspired by research on superposition in neural networks, particularly:

Elhage et al. (2022). *Toy Models of Superposition*. Transformer Circuits.

https://transformer-circuits.pub/2022/toy_model/index.html
