# Predicting Online Shoppers' Purchase Intent
**Iris Hoxha · Ina Osmeni · Ina Shametaj**
University of New York Tirana — Machine Learning course project

---

## What this is

We built a two-phase ML pipeline to predict whether a browsing session on an e-commerce site will end in a purchase. The dataset is from UCI (Sakar et al. 2019) — 12,330 sessions from a real online store, recorded over a full year. Only 15.5% of sessions actually convert, so basically every decision we made was shaped by that imbalance.

The project is split into two notebooks that are meant to be run in order.

---

## Files

```
phase1_eda.ipynb EDA, cleaning, feature engineering, SMOTE
phase2_models.ipynb Model training and evaluation
paper_onlineshopping.tex Full paper (LaTeX, shared on Overleaf)
Paper_onlineshopping.pdf Compiled PDF of the paper
Phase2_Presentation.pptx Presentation slides
ML_Phase2_Presentation.pptx Presentation slides (repaired version)
README.md This file

X_train_scaled_smote.csv Training features after scaling + SMOTE
y_train_smote.csv
X_train_scaled.csv Training features after scaling (no SMOTE)
y_train.csv
X_test_scaled.csv Test features (never touched after split)
y_test.csv
```

---

## Phase 1 — EDA and data prep

The first notebook does all the groundwork before any modelling. The main steps were:

1. Removed 125 duplicate rows
2. Type conversions (Revenue and Weekend were stored as strings, Month as abbreviations)
3. Log transforms on the skewed duration/PageValues features, winsorisation on page counts
4. Built 15 new session-level features — the most useful ones ended up being `SessionValueScore` (PageValues × (1 − ExitRates)) and `HasPageValue`
5. Feature selection: dropped highly correlated pairs first, then took the union of top-15 from mutual information and random forest importance → **20 final features**
6. Stratified 80/20 train/test split
7. StandardScaler fit only on training data
8. SMOTE on the training set to balance classes (ended up with 16,476 samples)

The 6 CSV files produced here are what Phase 2 loads.

---

## Phase 2 — Models

We trained three models: Logistic Regression, SVM with RBF kernel, and a feedforward Neural Network in Keras. Tuned with GridSearchCV (5-fold for LR, 3-fold for SVM because it was already taking forever, manual grid for NN with early stopping on val AUC).

**Best configs:**

| Model | Best settings |
|-------|--------------|
| Logistic Regression | C=0.01, L2, no class weighting |
| SVM | C=10, γ=0.1, class_weight=balanced |
| Neural Network | layers=(128,64), dropout=0.3, lr=0.001 |

**Test set results (2,441 sessions):**

| | Logistic Reg. | SVM | Neural Net |
| -- | :--: | :--: | :--: |
| Accuracy | 0.8787 | 0.8492 | 0.8759 |
| Precision | 0.5817 | 0.5132 | 0.5741 |
| Recall | 0.8010 | 0.7120 | 0.8010 |
| **F1** | **0.6740** | 0.5965 | 0.6689 |
| ROC-AUC | 0.9185 | 0.8837 | 0.9227 |
| **PR-AUC** | 0.6656 | 0.5762 | **0.7160** |

F1 and PR-AUC are the metrics that actually matter here — accuracy is useless when 84.5% of sessions are the same class.

---

## What we found

- LR and the NN are basically tied on F1 (0.674 vs 0.669). We think it's because Phase 1 already encoded most of the non-linear structure as new features, so there wasn't much left for the NN to discover.
- The NN does win on PR-AUC by a real margin (0.716 vs 0.666), which matters if you care about ranking sessions rather than just binary prediction at 0.5.
- SVM came last on everything despite having the highest CV score (0.912). We're pretty sure it overfit to the SMOTE-generated synthetic examples — its test F1 was 0.597.
- We applied SMOTE before CV which caused some score inflation. It doesn't affect the test results but the hyperparameter selection might be slightly off because of it.

---

## How to run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow imbalanced-learn
```

Run `phase1_eda.ipynb` first — it generates the 6 CSV files. Then run `phase2_models.ipynb`. SVM training will take a few minutes.

---

## Paper

The full writeup is `paper_onlineshopping.tex`, shared on Overleaf. The compiled PDF (`Paper_onlineshopping.pdf`) is also in the repo.

The project is shared directly on Overleaf — no upload needed.

## Presentation

Slides are in `Phase2_Presentation.pptx` and `ML_Phase2_Presentation.pptx`.

---

## Limitations

- SMOTE was applied before cross-validation (leakage — CV scores are slightly inflated)
- Random train/test split, not time-ordered, so seasonal effects aren't handled properly
- Everything evaluated at the default 0.5 threshold, not calibrated to actual cost
- Single e-commerce site, single year — might not generalise
