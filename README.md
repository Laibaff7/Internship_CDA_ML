Cassini CDA Dust Analysis

This repository contains the machine learning pipeline and analysis for the Cosmic Dust Analyzer (CDA) mass spectrometry data. It includes unsupervised learning experiments, VQ-VAE architecture implementation, and hyperparameter optimization.

Directory Structure

📂 Notebooks_ML

Contains Jupyter notebooks and scripts focused on labeled unsupervised learning. This section explores the clustering and feature extraction of manually classified spectral data.


📂 Scripts_ML

Houses the core Python implementation for the VQ-VAE (Vector Quantized-Variational Autoencoder) architecture. This is utilized for the unlabeled portion of the dataset to learn robust latent representations.


📂 optuna_study_20260310_211416

Contains the results of hyperparameter optimization performed via Optuna. This folder includes the optimal metrics and configuration parameters found for the VQ-VAE model.

📂 unknown_vqvae_analysis_latest

Includes the final analysis results for unknown spectral types. This directory provides:

The final model paths and weights.

Evaluation results and reconstruction analysis.


Latent Space Explorer

To interactively visualize and explore the learned latent space, the Latent Explorer tool is hosted on Hugging Face:


🔗 Hugging Face Profile: Laibaff7
