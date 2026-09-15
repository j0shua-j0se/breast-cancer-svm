# Breast Cancer Classification — Support Vector Machines

Task 7 of the Elevate Labs AI & ML internship (Sep–Nov 2025).

## Overview
Binary classification of the Breast Cancer Wisconsin dataset (Kaggle, via kagglehub) with linear and RBF SVMs, PCA decision-boundary visualization, and `GridSearchCV` hyperparameter tuning.

## Files
- `Task_7.ipynb` — notebook with the full pipeline
- `breast-cancer.csv` — input data (569 rows, 30 numeric features; target `diagnosis`, M=malignant/B=benign)

## What it does
1. Download the dataset with `kagglehub` and load it (drop `id`, label-encode `diagnosis`)
2. Stratified train/test split (80/20, `random_state=42`) and `StandardScaler`
3. Train a linear-kernel SVM and an RBF-kernel SVM; report accuracy, confusion matrix, classification report for each
4. Visualize decision boundaries in 2D via PCA
5. Tune `C` and `gamma` for the RBF kernel with 5-fold `GridSearchCV` (scoring = accuracy)
6. Run 5-fold stratified cross-validation with the best model

## Results
Exact metrics printed in the notebook (test set, n=114):
- Linear SVM accuracy: `0.9649`
- RBF SVM accuracy: `0.9737`
- GridSearchCV best hyperparameters: `{'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}`
- GridSearchCV best CV accuracy: `0.9758241758241759`
- Best SVM test accuracy: `0.9737`
- 5-fold CV accuracy (best model, full dataset): `0.9754 ± 0.0195`

Note: `GridSearchCV` is scored on accuracy, not ROC-AUC — the notebook does not compute an ROC-AUC score, so an "SVM ROC-AUC after GridSearchCV" figure is not printed anywhere in this notebook.

## How to run
Open `Task_7.ipynb` in Jupyter and run all cells. The notebook reads from a hardcoded local `INPUT_CSV`; the first cell downloads the dataset via `kagglehub`, so update the path to match its printed location if needed.

Requires: pandas, numpy, seaborn, matplotlib, scikit-learn, kagglehub

## Author
Joshua Jose · MSc Data Science, FAU Erlangen-Nürnberg · linkedin.com/in/j0shuaj0se
