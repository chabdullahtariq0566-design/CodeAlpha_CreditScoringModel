# 💳 Task 1 — Credit Scoring Model
### CodeAlpha Machine Learning Internship

---

## 📌 Objective
Predict whether an individual is **creditworthy** or not based on their past financial history using supervised machine learning classification algorithms.

---

## 🧠 Problem Statement
Banks and financial institutions need to assess the risk of lending money to individuals. A credit scoring model automates this decision by analyzing financial behavior patterns such as income, debt levels, payment history, and credit score — then predicting the likelihood of default.

---

## 📂 Dataset
| Source | Details |
|--------|---------|
| **Primary** | German Credit Dataset — `uciml/german-credit` on Kaggle |
| **Fallback** | Synthetic dataset (auto-generated, always works) |

### Features Used
| Feature | Description |
|---------|-------------|
| `Age` | Age of the individual |
| `Income` | Annual income (USD) |
| `LoanAmount` | Total loan requested |
| `LoanTerm` | Repayment period (months) |
| `EmploymentYears` | Years at current job |
| `NumCredits` | Number of existing credit lines |
| `MissedPayments` | Number of missed payments |
| `DebtRatio` | Debt-to-asset ratio |
| `CreditScore` | Credit bureau score (300–850) |
| `HasMortgage` | Owns a mortgage (0/1) |
| `HasSavings` | Has savings account (0/1) |

### Engineered Features
- `DebtToIncome` — Loan amount relative to income
- `MonthlyPayment` — Estimated monthly repayment
- `PaymentToIncome` — Monthly payment vs monthly income
- `CreditUtilization` — Debt ratio scaled by number of credits
- `RiskScore` — Composite risk indicator

### Target Variable
- `Creditworthy` → **1** = Low risk (approve) | **0** = High risk (reject)

---

## ⚙️ Approach & Methodology

```
Raw Data
   ↓
Exploratory Data Analysis (distributions, correlations)
   ↓
Feature Engineering (5 new features)
   ↓
Train / Test Split (80/20, stratified)
   ↓
StandardScaler (for Logistic Regression)
   ↓
Model Training × 4 algorithms
   ↓
5-Fold Stratified Cross-Validation
   ↓
Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
   ↓
Visualisation (ROC curves, confusion matrix, feature importance)
```

---

## 🤖 Models Used

| Model | Key Hyperparameters |
|-------|-------------------|
| **Logistic Regression** | C=1.0, max_iter=1000 |
| **Decision Tree** | max_depth=6 |
| **Random Forest** | n_estimators=200, max_depth=8 |
| **Gradient Boosting** | n_estimators=200, lr=0.08, max_depth=4 |

---

## 📊 Evaluation Metrics
- **Accuracy** — Overall correct predictions
- **Precision** — Of predicted creditworthy, how many truly are
- **Recall** — Of truly creditworthy, how many are caught
- **F1-Score** — Harmonic mean of Precision & Recall
- **ROC-AUC** — Area under the ROC curve (main metric)
- **CV-AUC** — 5-fold cross-validated AUC (generalization check)

---

## 📈 Expected Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~0.80 | ~0.86 |
| Decision Tree | ~0.78 | ~0.82 |
| Random Forest | ~0.85 | ~0.91 |
| Gradient Boosting | ~0.86 | ~0.92 |

> Best model: **Gradient Boosting / Random Forest**

---

## 📦 Output Files Generated
- `eda_distributions.png` — Feature distribution histograms
- `correlation_heatmap.png` — Feature correlation matrix
- `task1_results.png` — Metrics comparison, ROC curves, confusion matrix
- `feature_importance.png` — Random Forest feature importances

---

## 🚀 How to Run on Kaggle

1. Go to [kaggle.com](https://kaggle.com) → **+ New Notebook**
2. *(Optional)* Add dataset: **+ Add Data** → search `uciml/german-credit`
3. Paste `task1_credit_scoring.py` content into a cell
4. Click **Run All**
5. No GPU needed — runs on CPU in ~1–2 minutes

---

## 🛠️ Libraries Required
```
numpy, pandas, matplotlib, seaborn, scikit-learn
```
All pre-installed on Kaggle.

---

## 📁 GitHub Repository
Name your repo: `CodeAlpha_CreditScoringModel`

---

*CodeAlpha Machine Learning Internship — Task 1 of 4*
