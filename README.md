# Cognitive Neural Conflict Processing using Graphical Models

This is the GitHub repo for the final project for 10708 (Probabilistic Graphical Models) by Jacob Yeung and Yuxin Guo.

This repository contains the code required to reproduce our results.

## Data Preprocessing
We include the script for preprocessing the neural data in `data_preprocess.py`.

##  Logistic Regression Baseline
We include the notebook for running the logistic regression baseline models in `logistic_regression.ipynb`. 

## Hiearchical Latent Variable Models
We include the notebook for running the hierarchicial latent variable models in `hlvm.ipynb`. 

## Causal Latent Models
We include the code for running the causal latent models in the directory `causal_latent_analysis`.
For training the generative causal explainer, run neural_data_analysis.py. To evaluate the classification accuracies, run reconstruct_classification.py. To analyze the effect of modifying the causal and non-causal latent variables, run perturb_latents.py. These files were used for generating results (Fig.3 and Fig.4) in Section 4.3.2. 