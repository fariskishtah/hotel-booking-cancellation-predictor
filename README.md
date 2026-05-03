# 🏨 Hotel Booking Cancellation Prediction

> **EED472 — Machine Learning and Pattern Recognition**  
> Future University in Egypt · Spring 2026  
> Team Size: 3 students · Project Weight: 10%

---

## 📋 Project Overview

This project applies supervised machine learning to predict whether a hotel booking will be **cancelled before the arrival date**, using the publicly available Hotel Booking Demand dataset (119,390 bookings across City and Resort hotels).

**Target:** `is_canceled` (Binary Classification — 0 = Kept, 1 = Cancelled)

---

## 📁 Project Structure

```
hotel_ml_project/
│
├── data/
│   └── hotel_bookings.csv          ← Raw dataset (119,390 rows × 32 features)
│
├── src/
│   ├── data_loader.py              ← Data loading and initial validation
│   ├── preprocessing.py            ← Cleaning, feature engineering, sklearn Pipeline
│   ├── eda.py                      ← 10 EDA/KPI visualisations
│   ├── train.py                    ← Train 5 models + hyperparameter tuning
│   ├── evaluate.py                 ← Evaluation plots + model comparison
│   └── pipeline.py                 ← ★ Master script (runs all steps)
│
├── models/
│   ├── best_model.pkl              ← Saved best model (LightGBM)
│   ├── feature_names.pkl           ← Feature list
│   ├── model_results.csv           ← All model metrics
│   └── feature_importance.csv      ← Feature importance scores
│
├── outputs/
│   └── figures/                    ← All generated plots (PNG)
│
├── assets/
│   └── logo.png                     ← Streamlit sidebar logo
│
├── report/
│   └── final_report.md             ← Full academic report
│
├── presentation/
│   └── outline.md                  ← 10-minute presentation outline
│
├── app.py                          ← ★ Streamlit web application
├── requirements.txt                ← Python dependencies
├── packages.txt                    ← Linux dependency for cloud deployment
└── README.md                       ← This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
cd hotel_ml_project
python src/pipeline.py
```

This will:
1. Load and clean the dataset
2. Engineer new features
3. Generate 10 EDA visualisations → `outputs/figures/`
4. Train 5 ML models with cross-validation
5. Tune the best model (LightGBM)
6. Generate evaluation plots

### 3. Launch the Streamlit App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🌍 Public Web App Deployment

The recommended setup is:

1. Upload this project to a **public GitHub repository**.
2. Deploy `app.py` from that repository on **Streamlit Community Cloud**.
3. Share the generated `streamlit.app` link with your doctor and classmates.

On Streamlit Community Cloud:

- Repository: your public GitHub repo
- Branch: `main`
- Main file path: `app.py`
- Python version: choose `3.11` in **Advanced settings**

After deployment, viewers can open the app link directly in a browser. They do not need to install Python or run anything from the terminal.

---

## 📊 Dataset

| Attribute | Value |
|---|---|
| Source | [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) |
| Original size | 119,390 rows × 32 features |
| After cleaning | 87,230 rows |
| Target | `is_canceled` (0/1) |
| Class balance | 63% not cancelled / 37% cancelled |

---

## 🔧 Data Cleaning Steps

1. **Remove duplicates** — 31,994 exact duplicate rows removed
2. **Handle nulls** — Agent/Company → 0; Country → "Unknown"; Children → 0
3. **Remove zero-guest rows** — 166 impossible bookings removed
4. **Fix "Undefined" categoricals** — Meal → "SC"; Market Segment → "Online TA"; Distribution Channel → "TA/TO"
5. **ADR outlier clipping** — IQR-based clipping (2,508 outliers)
6. **Type conversions** — Children/Agent → int; Dates → datetime

---

## ⚙️ Feature Engineering

| Feature | Description |
|---|---|
| `total_nights` | Weekend + week nights |
| `total_guests` | Adults + children + babies |
| `room_upgraded` | Reserved ≠ assigned room type |
| `has_children` | Binary: booking with minor children |
| `revenue_per_night` | ADR × total nights |
| `is_long_stay` | total_nights > 7 |
| `month_num` | Month as integer (1–12) |
| `season` | Winter / Spring / Summer / Autumn |
| `cancel_rate_history` | Previous cancellations / all previous bookings |

---

## 🤖 Models

| Model | Family | Test ROC-AUC | Test F1 |
|---|---|---|---|
| Logistic Regression | Linear | 0.8185 | 0.5021 |
| Decision Tree | Tree | 0.8852 | 0.6659 |
| Random Forest | Ensemble (Bagging) | 0.9085 | 0.6763 |
| XGBoost | Ensemble (Boosting) | 0.9116 | 0.7032 |
| **LightGBM** ✅ | **Ensemble (Boosting)** | **0.9122** | **0.7035** |

---

## 🏆 Best Model — LightGBM

```
Accuracy  : 84.86%
Precision : 77.62%
Recall    : 64.06%
F1 Score  : 70.35%
ROC-AUC   : 91.22%
```

### Top Predictive Features
1. `lead_time` — Longer lead time → higher cancellation risk
2. `total_of_special_requests` — More requests → less likely to cancel
3. `adr` — Price sensitivity
4. `deposit_type` — Non-refund deposits cancel at nearly 100%
5. `cancel_rate_history` — Historical behaviour predicts future

---

## 🌐 Streamlit App Features

| Tab | Contents |
|---|---|
| 📊 Dashboard | Interactive EDA: KPIs, monthly trends, market segments, lead time |
| 🔮 Predictor | Enter booking details → get cancellation probability + gauge |
| 📈 Model Results | Performance table, radar chart, feature importance |
| ℹ️ About | Project description and architecture |

---

## 📚 References

- Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49.
- Scikit-learn documentation: https://scikit-learn.org
- LightGBM documentation: https://lightgbm.readthedocs.io
- XGBoost documentation: https://xgboost.readthedocs.io

---

## ⚖️ Ethical Considerations

- Dataset is anonymised — no personally identifiable information
- Country codes preserved for analytics only
- Model should not be used for discriminatory pricing
- Public dataset from Kaggle (CC0 Public Domain)
