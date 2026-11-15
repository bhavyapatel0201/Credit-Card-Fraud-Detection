# Credit Card Fraud Detection

Machine learning project that detects fraudulent credit card transactions in **highly imbalanced data**, where fraud makes up only a tiny fraction of all transactions.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## Problem

Fraudulent transactions are rare, so a model that predicts "not fraud" for everything can look accurate while catching no fraud at all. This project focuses on handling the class imbalance and judging models with metrics that matter for fraud (precision, recall, F1, ROC-AUC) instead of plain accuracy.

## Dataset

- **Source:** [FILL IN: dataset name and link, e.g. Kaggle "Credit Card Fraud Detection"]
- **Size:** [FILL IN: number of transactions, fraud percentage]
- The raw data file is not stored in this repository (large CSVs are excluded in `.gitignore`). Download it from the link above and place it in a `data/` folder.

## Approach

1. **Exploration** (`01_explore.ipynb`): inspect the data, check the class balance, and look at feature distributions.
2. **Handling imbalance:** [FILL IN: e.g. class weights, SMOTE, undersampling]
3. **Models:** [FILL IN: e.g. logistic regression, random forest]
4. **Evaluation:** precision, recall, F1-score, confusion matrix and ROC-AUC on a held-out test set.

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| [FILL IN] | | | | |

[FILL IN: one or two sentences on the best model and what the numbers mean.]

## Getting started

```bash
git clone https://github.com/bhavyapatel0201/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter notebook 01_explore.ipynb
```

## Repository structure

```
├── 01_explore.ipynb    # Data exploration and modelling
├── requirements.txt    # Python dependencies
└── .gitignore
```

## Author

**Bhavya Patel**, Master of Applied Computing, University of Windsor
[LinkedIn](https://linkedin.com/in/bhavya-patel-data) · [Portfolio](https://bhavyapatel0201.github.io)
