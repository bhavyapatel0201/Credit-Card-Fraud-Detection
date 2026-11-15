# Credit Card Fraud Detection

Machine learning project that detects fraudulent credit card transactions in **highly imbalanced data**, where fraud makes up only a tiny fraction of all transactions.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## Problem

Fraudulent transactions are rare, so a model that predicts "not fraud" for everything can look accurate while catching no fraud at all. In this dataset that lazy model would score about 99.8% accuracy and still miss every fraudulent transaction.

This project focuses on handling the class imbalance and on judging models with metrics that reflect the real cost of fraud (precision, recall, F1, ROC-AUC) instead of plain accuracy.

## Dataset

- **Source:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), published on Kaggle by the ULB Machine Learning Group.
- **Size:** 284,807 transactions made by European cardholders over two days in September 2013, of which **492 are fraud (0.172%)**.
- **Features:** `Time`, `Amount`, and `V1` to `V28`, which are anonymised principal components of the original transaction data. The target is `Class` (1 = fraud, 0 = legitimate).
- The raw data file is not stored in this repository (large CSVs are excluded in `.gitignore`). Download `creditcard.csv` from the link above and place it in a `data/` folder.

## Approach

1. **Exploration** (`01_explore.ipynb`): inspect the data, check the class balance, and look at feature distributions.
2. **Preprocessing:** a stratified 80/20 train/test split so both sets keep the same fraud ratio, and feature scaling with `StandardScaler`.
3. **Handling imbalance:** compare class weighting with resampling. Resampling is applied to the training data only, so no information leaks into the test set.
4. **Models:** logistic regression as a baseline (plain, on scaled features, with class weights, and on SMOTE-resampled data), compared with XGBoost.
5. **Evaluation:** precision, recall and F1 for the fraud class on a held-out test set that was never resampled, plus a precision-recall curve for choosing a decision threshold.

## Results

| Approach | Precision | Recall | F1 |
|---|---|---|---|
| Logistic regression (baseline) | 0.835 | 0.673 | 0.746 |
| Logistic regression, scaled features | 0.827 | 0.633 | 0.717 |
| Logistic regression, class weights | 0.061 | 0.918 | 0.114 |
| Logistic regression, SMOTE | 0.058 | 0.918 | 0.109 |
| **XGBoost** | **0.897** | **0.796** | **0.843** |
| XGBoost, threshold tuned on the precision-recall curve | 0.579 | 0.827 | 0.681 |

XGBoost gave the best balance (F1 0.843). Class weighting and SMOTE lift recall above 0.91 but drop precision to about 0.06, so only about 1 in 16 alerts would be real fraud. The tuned-threshold row was chosen on the same test set, so treat it as illustrative rather than an unbiased estimate.

Reading the table: **recall** is the share of real fraud the model catches, and **precision** is the share of its alerts that turn out to be real fraud. A fraud team usually wants high recall, but low precision floods analysts with false alarms, so F1 and the precision-recall trade-off matter more than accuracy.

## Possible next steps

- Choose the decision threshold on a separate validation split (here it was tuned on the test set), balancing missed fraud against false alarms.
- Use cross-validation to check the results are stable across splits.
- Add model explanations (for example SHAP) to show which features drive each alert.
- Wrap the best model in a small API so transactions can be scored in real time.

## Getting started

```bash
git clone https://github.com/bhavyapatel0201/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook 01_explore.ipynb
```

## Repository structure

```
├── 01_explore.ipynb    # Data exploration and modelling
├── requirements.txt    # Python dependencies
├── data/               # Place creditcard.csv here (not tracked)
└── .gitignore
```

## Author

**Bhavya Patel**, Master of Applied Computing, University of Windsor
[LinkedIn](https://linkedin.com/in/bhavya-patel-data) · [Portfolio](https://bhavyapatel0201.github.io)
