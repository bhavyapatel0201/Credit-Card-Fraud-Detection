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
2. **Preprocessing:** a stratified train/test split so both sets keep the same fraud ratio, and scaling of `Amount` and `Time`.
3. **Handling imbalance:** compare class weighting with resampling. Resampling is applied to the training data only, so no information leaks into the test set.
4. **Models:** a logistic regression baseline compared with tree-based ensemble models.
5. **Evaluation:** precision, recall, F1-score, confusion matrix and ROC-AUC on a held-out test set that was never resampled.

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic regression (baseline) | | | | |
| Tree-based ensemble | | | | |

Reading the table: **recall** is the share of real fraud the model catches, and **precision** is the share of its alerts that turn out to be real fraud. A fraud team usually wants high recall, but low precision floods analysts with false alarms, so F1 and the precision-recall trade-off matter more than accuracy.

## Possible next steps

- Tune the decision threshold for a chosen balance between missed fraud and false alarms.
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
