# RT-IoT 2022 Intrusion Detection — WCNS Project Lab

A reproducible machine-learning pipeline for multi-class intrusion detection on the **RT-IoT 2022** dataset, implemented in a single Jupyter notebook: `WCNS Project Lab.ipynb`.  
This repository covers data loading, preprocessing (scaling, PCA), imbalance handling (SMOTE), model training/evaluation across several algorithms, and a concise results summary.

> **Notebook:** `WCNS Project Lab.ipynb`  
> **Dataset:** UCI Machine Learning Repository — RT-IoT 2022 (id=942), fetched via `ucimlrepo`

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Environment & Setup](#environment--setup)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models & Results](#models--results)
- [Key Findings](#key-findings)
- [Limitations & Next Steps](#limitations--next-steps)
- [Reproducibility](#reproducibility)
- [How to Run](#how-to-run)
- [Citations](#citations)
- [License](#license)

---

## Overview
This project builds and evaluates multiple classifiers for IoT network intrusion detection. It focuses on practical steps:
- Loading the RT-IoT 2022 dataset directly from UCI via `ucimlrepo`.
- Cleaning and preparing the data.
- Addressing class imbalance with **SMOTE**.
- Reducing dimensionality with **PCA** while retaining ≈95% variance (→ 26 components).
- Training and evaluating several models: **KNN**, **Naïve Bayes**, **SVM**, and an **Artificial Neural Network (MLPClassifier)**.
- Reporting accuracy, precision, recall, F1-score, and ROC-AUC.

---

## Dataset
- **Source:** UCI Machine Learning Repository — *RT-IoT 2022* (id = 942)
- **Initial shape:** 123,117 rows × 84 columns  
- **After removing duplicates:** 117,922 × 84
- **Target column:** `Attack_type` (multiclass, highly imbalanced)

> Major classes include `DOS_SYN_Hping`, `Thing_Speak`, `ARP_poisioning`, `MQTT_Publish`, various `NMAP_*` scans, etc.

---

## Environment & Setup
Recommended Python ≥ 3.10.

```bash
# Create & activate a virtual environment (venv as example)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# Install core dependencies
pip install -U pip
pip install numpy pandas scikit-learn imbalanced-learn matplotlib ucimlrepo jupyter
```

> If you plan to export or persist models, also install: `joblib`

---

## Project Structure
```
.
├── WCNS Project Lab.ipynb     # Main analysis notebook
└── README.md                  # You are here
```

---

## Methodology
1. **Data Loading** — `ucimlrepo.fetch_ucirepo(id=942)` retrieves features/labels.
2. **Preprocessing**
   - **Scaling**: `StandardScaler` on numeric features.
   - **Dimensionality Reduction**: `PCA(n_components=26)` to capture ~95% variance.
3. **Train/Test Split**
   - `train_test_split(..., test_size=0.30, random_state=42, stratify=y)`
4. **Imbalance Handling**
   - **SMOTE** applied **only on the training set** to synthesize minority-class samples.
5. **Models**
   - **K-Nearest Neighbors (KNN)**
   - **Naïve Bayes**
   - **Support Vector Machine (SVM)**
   - **Artificial Neural Network (MLPClassifier)**
6. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1 (weighted), ROC-AUC (one-vs-rest).
   - Classification reports by class.

> **Note on best practice**: To avoid any chance of data leakage, consider wrapping `StandardScaler → PCA → SMOTE → Classifier` in an **`imblearn.pipeline.Pipeline`** and fitting only on the training data (optionally within **Stratified K-Fold CV**).

---

## Models & Results
*(Values copied from the latest saved notebook run; “time” is approximate execution time on the referenced machine.)*

| Model | Test Accuracy | F1 (weighted) | Precision | Recall | ROC-AUC (OvR) | Train Accuracy | Exec Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| K-Nearest Neighbors | 0.9967 | 0.9967 | 0.9907 | 0.9903 | 0.9953 | 0.9998 | 6065.37 |
| Naïve Bayes | 0.8967 | 0.9069 | 0.9316 | 0.8918 | 0.9882 | 0.7329 | 6.53 |
| Support Vector Machine | 0.9611 | 0.9612 | 0.9662 | 0.9686 | 0.9967 | 0.9185 | 62.11 |
| Artificial Neural Network | 0.9956 | 0.9957 | 0.9904 | 0.9902 | 0.9977 | 0.9959 | 664.93 |

**Observations**
- **ANN (MLP)** and **KNN** achieve the **highest test accuracy/F1**.  
- **KNN** is **very slow** at scale; **ANN** offers a better performance–speed tradeoff.  
- **SVM** remains competitive with strong ROC-AUC.  
- **Naïve Bayes** is a fast baseline but trails in accuracy/F1.

---

## Key Findings
- Proper **SMOTE** usage improves minority-class detection, but extreme imbalance persists.
- **PCA** to 26 components maintains high accuracy while reducing computation.
- **ANN** is a strong candidate for deployment given its balance of performance and runtime versus KNN.

---

## Limitations & Next Steps
- **Imbalance**: Evaluate **macro-F1** and **balanced accuracy**, not only weighted metrics dominated by majority classes.
- **Data Leakage Risk**: Ensure scaling and PCA are **inside a pipeline** fit only on training folds.
- **Cross-Validation**: Add **Stratified K-Fold CV** to report mean±std metrics for robustness.
- **Diagnostics**: Add **confusion matrices** and **per-class error analysis**.
- **Modeling**: Benchmark **Random Forest / Gradient Boosting**; tune hyperparameters (with early stopping where applicable).
- **Deployment**: Export the selected model with `joblib` and include an inference example.

---

## Reproducibility
- Use a fixed `random_state` (e.g., `42`) for `train_test_split`, SMOTE, and model initializations.  
- Record library versions:
  ```bash
  python -c "import sys, sklearn, imblearn, numpy, pandas; print(sys.version); print('sklearn', sklearn.__version__); print('imblearn', imblearn.__version__); print('numpy', numpy.__version__); print('pandas', pandas.__version__)"
  ```

---

## How to Run
```bash
# 1) Create environment & install deps (see above)
# 2) Launch Jupyter
jupyter notebook

# 3) Open and run all cells in: WCNS Project Lab.ipynb
```

### Optional: Quick Inference Sketch
```python
# After training and saving your best pipeline as 'model.joblib':
import joblib
import pandas as pd

pipe = joblib.load('model.joblib')
X_new = pd.read_csv('sample_features.csv')  # same feature schema used in training
preds = pipe.predict(X_new)
probs = pipe.predict_proba(X_new)  # if supported
```

---

## Citations
- UCI Machine Learning Repository: **RT-IoT 2022** (id=942)
- SMOTE: N. V. Chawla et al., “SMOTE: Synthetic Minority Over-sampling Technique,” *JAIR*, 2002.
- scikit-learn, imbalanced-learn documentation.

---

## License
This project is released under the **MIT License** (adjust if your org/course requires a different license).