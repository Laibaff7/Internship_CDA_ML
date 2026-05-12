# Cassini CDA Dust Analysis

This repository contains the machine learning pipeline and analysis for the **Cosmic Dust Analyzer (CDA)** mass spectrometry data. It includes unsupervised learning experiments, VQ-VAE architecture implementation, and hyperparameter optimization.

---

## 📂 Project Structure

### **`Notebooks_ML/`**
Contains Jupyter notebooks and scripts focused on **labeled unsupervised learning**. This section explores the clustering and feature extraction of manually classified spectral data.

### **`Scripts_ML/`**
Houses the core Python implementation for the **VQ-VAE (Vector Quantized-Variational Autoencoder)** architecture. This is utilized for the unlabeled portion of the dataset to learn robust latent representations.

### **`optuna_study_20260310_211416/`**
Contains the results of hyperparameter optimization performed via **Optuna**. This folder includes:
* **Optimal metrics** for the VQ-VAE.
* Configuration parameters for the best-performing models.

### **`unknown_vqvae_analysis_latest/`**
Includes the final analysis results for **unknown spectral types**. This directory provides:
* **Model Paths:** The final saved weights and model states.
* **Results:** Evaluation metrics and reconstruction analysis.

---

## Latent Space Explorer

To interactively visualize and explore the learned latent space, the **Latent Explorer** tool is hosted on Hugging Face.

### **🔗 [Hugging Face Profile: Laibaff7](https://huggingface.co/Laibaff7)**


