# Loan Approval and Maximum Loan Prediction

A machine learning project that addresses two related lending problems: predicting whether a loan application will be approved (classification), and predicting the maximum loan amount that can be safely offered to approved applicants (regression).

## Project Overview

| | Case Study A | Case Study B |
|---|---|---|
| **Task** | Classification | Regression |
| **Target** | Loan approval status (Approved / Rejected) | Maximum allowed loan amount (£) |
| **Dataset** | 58,642 cleaned records | 50,292 approved-loan records |
| **Best Model** | KNN (K=7, Manhattan) / Voting Ensemble | Pruned Decision Tree (depth=4) |

## Dataset

**Source file:** `loan_approval_data.csv` — 58,645 rows × 13 columns

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Applicant age (capped at 80) |
| `income` | Numeric | Annual income in GBP |
| `employment_length` | Numeric | Years of employment (capped at 50) |
| `loan_intent` | Categorical | EDUCATION, MEDICAL, HOMEIMPROVEMENT, DEBTCONSOLIDATION, PERSONAL, VENTURE |
| `loan_amount` | Numeric | Requested loan amount (£500–£35,000) |
| `loan_interest_rate` | Numeric | Annual interest rate (%) |
| `loan_income_ratio` | Numeric | Loan-to-income ratio |
| `payment_default_on_file` | Binary | Prior payment default (Y/N) |
| `credit_history_length` | Numeric | Years of credit history |
| `home_ownership` | Categorical | MORTGAGE, OWN, RENT, OTHER |
| `loan_approval_status` | Target (A) | 0 = Approved, 1 = Rejected |
| `max_allowed_loan` | Target (B) | Maximum safe loan amount in GBP |

## Notebooks

| Notebook | Description |
|---|---|
| `Data Cleaning NB1.ipynb` | Data cleaning, outlier handling, missing value imputation, encoding |
| `Classification NB2.ipynb` | Naive Bayes, Logistic Regression, KNN — comparison and hyperparameter tuning |
| `Regression and Ensemble NB3.ipynb` | Voting ensemble classifier and Decision Tree regression models |

Run them in order (NB1 → NB2 → NB3).

## Methodology

### Data Cleaning (NB1)

- **Missing values:** Median imputation for `age` and `loan_interest_rate`; mode imputation for `payment_default_on_file`
- **Outlier capping:** `age` → 80, `employment_length` → 50 years, `income` → 3×IQR fence (~£176,400)
- **Invalid records:** 3 rows with negative `max_allowed_loan` removed
- **Encoding:** Label encoding for `payment_default_on_file`; one-hot encoding for `home_ownership` and `loan_intent`

### Classification (NB2)

Three models were trained and compared on an 80/20 stratified train-test split:

| Model | Accuracy | Recall | Precision | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Naive Bayes | 88.51% | 31.74% | 71.82% | 0.440 | 0.845 |
| Logistic Regression | 89.68% | 42.69% | 73.81% | 0.541 | 0.889 |
| **KNN (K=5)** | **91.77%** | **54.61%** | **81.50%** | **0.654** | 0.863 |

KNN was selected as the best individual model by F1-Score. GridSearchCV tuning (K ∈ {3,5,7,11,15,21}, distance ∈ {euclidean, manhattan}) yielded **K=7, manhattan distance** as optimal, improving F1 to 0.665 and AUC to 0.876.

### Ensemble & Regression (NB3)

**Voting Ensemble (soft voting — Logistic Regression + KNN):**

| Metric | Score |
|---|---|
| Accuracy | 91.83% |
| Recall | 51.08% |
| Precision | 85.81% |
| F1-Score | 0.640 |
| AUC-ROC | **0.911** |

The ensemble achieved the highest AUC-ROC across all classifiers.

**Decision Tree Regression:**

| Model | Depth | Leaves | MSE | MAE | R² |
|---|---|---|---|---|---|
| DT-1 (unpruned) | 24 | 23,041 | £1.62B | £2,713 | 0.531 |
| **DT-2 (pruned)** | **4** | **16** | **£1.48B** | £12,047 | **0.573** |

The pruned tree generalises better (higher R²) by avoiding overfitting.

## Results Summary

- **Best classifier (individual):** KNN with K=7 and Manhattan distance — F1: 0.665, AUC: 0.876
- **Best classifier (ensemble):** Soft-voting (LR + KNN) — AUC: 0.911
- **Best regression model:** Pruned Decision Tree (depth=4) — R²: 0.573
- **Example prediction (Client 60256):** 56-year-old, £57k income, £25.7k medical loan → maximum loan ≈ £88,000

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Usage

```bash
# Clone the repository
git clone https://github.com/mathushihanratnatheepan/loan-approval-and-max-loan-prediction.git
cd loan-approval-and-max-loan-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Launch Jupyter and run notebooks in order
jupyter notebook
```

Open and run the notebooks in this order:
1. `Data Cleaning NB1.ipynb`
2. `Classification NB2.ipynb`
3. `Regression and Ensemble NB3.ipynb`

## Project Structure

```
loan-approval-and-max-loan-prediction/
├── loan_approval_data.csv          # Raw dataset
├── Data Cleaning NB1.ipynb         # Preprocessing pipeline
├── Classification NB2.ipynb        # Classification models
├── Regression and Ensemble NB3.ipynb  # Ensemble + regression models
└── Analysis Report.pdf             # Full written report
```

## Author

**Mathushihan Rathnatheepan**
Module: Machine Learning & Data Mining (April 2025)
