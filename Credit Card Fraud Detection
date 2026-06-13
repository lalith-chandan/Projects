# Credit Card Fraud Detection — Neural Network vs. XGBoost

Detecting fraudulent credit-card transactions, and comparing two model families on the job: a small **neural network** and an **XGBoost** classifier. The whole project lives in one notebook that runs top to bottom — preprocessing, EDA, both models, and a final test-set comparison.

The short version: on this kind of **tabular, PCA-anonymised, highly imbalanced** data, the gradient-boosted trees come out ahead.

---

## Results on the held-out test set

56,962 transactions, 99 of them fraud. Each model is judged at the threshold it picked on the **validation** set — never on test.

| | Neural Net | XGBoost |
|---|---|---|
| **PR-AUC** | 0.75 | **0.87** |
| ROC-AUC | 0.98 | 0.97 |
| Frauds caught (of 99) | 76 | **83** |
| Precision | 0.84 | 0.83 |
| Recall | 0.77 | **0.84** |
| False alarms | 15 | 17 |

![Precision–Recall and ROC curves on the test set](images/test_pr_roc_curves.png)

![Confusion matrices on the test set](images/test_confusion_matrices.png)

The two models are basically tied on ROC-AUC (~0.97–0.98) — which is exactly the trap with imbalanced data. **PR-AUC tells the real story, and there XGBoost wins comfortably (0.87 vs 0.75)** while catching more frauds at comparable precision.

---

## Dataset

The well-known **ULB Credit Card Fraud Detection** dataset (European cardholders, September 2013).

- **284,807** transactions, **492** of them fraud — a **0.172%** positive rate (extreme imbalance).
- **30 input features:**
  - `Time` — seconds since the first transaction.
  - `V1`–`V28` — anonymised features from a **PCA** transformation of the original (confidential) details.
  - `Amount` — the transaction amount.
- **`Class`** — the label: `1` = fraud, `0` = legitimate.

> **The raw `creditcard.csv` is not included** (it's ~100 MB). Download it from Kaggle and drop it in the project root:
> https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
>
> The pre-split, pre-scaled data is provided under `data/` if you'd rather skip the preprocessing.

---

## Repository structure

```
.
├── credit_card_fraud_detection.ipynb   # The whole project, runs top to bottom
├── README.md
├── requirements.txt
├── .gitignore
├── images/                             # Figures used in this README
│   ├── correlation_heatmap.png
│   ├── nn_training_curves.png
│   ├── xgb_feature_importance.png
│   ├── test_pr_roc_curves.png
│   └── test_confusion_matrices.png
└── data/                               # Pre-split, pre-scaled datasets
    ├── train_scaled.csv
    ├── val_scaled.csv
    └── test_scaled.csv
```

---

## What the notebook does

**Split.** 60 / 20 / 20 into train / validation / test, **stratified** on the label so the ~0.172% fraud rate is preserved in all three sets.

| Set | Rows | Frauds | Fraud % |
|---|---|---|---|
| train | 170,884 | 295 | 0.173% |
| validation | 56,961 | 98 | 0.172% |
| test | 56,962 | 99 | 0.174% |

**Scale.** Only `Time` and `Amount` are standardized (the `V` features are already PCA outputs). The scaler is **fit on training data only** to avoid leaking information from the held-out sets. The scaled splits are saved to `data/`.

**Explore.** A correlation check confirms the `V` features are essentially uncorrelated with one another — a direct consequence of PCA producing orthogonal components. Only `Time` and `Amount` show meaningful correlations.

![Correlation heatmap](images/correlation_heatmap.png)

**Neural network.** Keras Tuner is used to sanity-check the architecture (number of layers and units); small, shallow networks do as well as anything, so the final model is a single hidden layer:

```
Input(30) → Dense(48, relu) → Dense(1, sigmoid)
```

Trained with `Adam(1e-3)`, **class weights** (~290× on fraud), early stopping and LR reduction on validation PR-AUC.

![Neural-net training curves](images/nn_training_curves.png)

**XGBoost.** Gradient-boosted trees with **`scale_pos_weight ≈ 578`** for the imbalance and early stopping on validation PR-AUC. It leans on a handful of PCA components — **V14, V12, V17, V10** — the same ones that stood out in EDA.

![XGBoost feature importance](images/xgb_feature_importance.png)

**Compare.** Both models are scored once on the test set, each at the threshold it chose on validation.

---

## Key takeaways

- **PCA features are uncorrelated by construction** — the near-zero correlation matrix is the signature of PCA orthogonality, not a quirk of the sample.
- **ROC-AUC hides the gap; PR-AUC reveals it.** Both models reach ~0.97–0.98 ROC-AUC, but XGBoost's 0.87 PR-AUC vs. the network's 0.75 is the honest comparison on imbalanced data.
- **Trees beat the neural net on tabular data.** With the features already reduced to PCA components, depth bought the network nothing, and XGBoost was both stronger and simpler to train.

---

## How to run

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# download creditcard.csv from Kaggle into the project root:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

jupyter notebook credit_card_fraud_detection.ipynb
```

Tested with **Python 3.10**. No GPU required.

---

## Possible next steps

- Calibrate the neural network's probabilities (its class-weighted scores aren't calibrated around 0.5).
- Tune XGBoost further, or try LightGBM / CatBoost.
- Use a cost-sensitive threshold based on the real cost of a missed fraud vs. a false alarm.

---
