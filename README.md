# PFairDP

This repo contains implementation of PFairDP, an automated pipeline for discovering 3D Pareto fronts between accuracy, privacy and fairness of ML models. The problem of finding the Pareto front is framed as an optimisation problem over hyperparameters (defined by all three objectives, e.g. number of epochs for neural network training or clipping norm for differential privacy). Then mutli-objective Bayesian optimisation is used.

The main dependencies are PyTorch, Opacus and AIF360.
