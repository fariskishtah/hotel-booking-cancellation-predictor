# EED472 Machine Learning Project — Final Report

**Title:** Hotel Booking Cancellation Prediction Using Supervised Machine Learning  
**Course:** EED472 — Machine Learning and Pattern Recognition  
**University:** Future University in Egypt, Department of Electrical Engineering  
**Semester:** Spring 2026  
**Team Members:** [Member 1 Name — ID], [Member 2 Name — ID], [Member 3 Name — ID]

---

## 1. Introduction & Motivation

The hospitality industry faces a significant operational challenge: booking cancellations. When guests cancel reservations — especially at the last minute — hotels incur direct revenue losses, inefficient resource allocation, and difficulties in demand forecasting. According to industry data, cancellation rates across European hotels range from 30–40%, with some online travel agency (OTA) bookings exceeding 50%.

Machine learning offers a data-driven approach to forecast cancellation risk at booking time, enabling proactive strategies such as targeted overbooking, dynamic pricing adjustments, and personalised retention offers.

**Research Question:** *Can we accurately predict whether a hotel booking will be cancelled, using features known at the time of booking?*

**Goal:** Build and compare multiple binary classification models that predict `is_canceled`, achieving ROC-AUC ≥ 0.85 on a held-out test set.

**Why It Matters:**
- Revenue optimisation: Hotels can overbook strategically to compensate for expected cancellations
- Resource planning: Staff scheduling, room preparation, and food ordering can be better planned
- Customer relationship management: High-risk bookings can receive targeted retention outreach

---

## 2. Dataset Description

### 2.1 Source
- **Name:** Hotel Booking Demand Dataset
- **Link:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- **Original paper:** Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49.

### 2.2 Overview

| Attribute | Value |
|---|---|
| Number of instances | 119,390 (original) → 87,230 (after cleaning) |
| Number of features | 32 (original) → 35 (after feature engineering) |
| Target variable | `is_canceled` (0 = Not Cancelled, 1 = Cancelled) |
| Problem type | Binary Classification |
| Hotel types | City Hotel (66.4%) and Resort Hotel (33.6%) |
| Date range | July 2015 – August 2017 |

### 2.3 Key Features

| Feature | Type | Description |
|---|---|---|
| `hotel` | Categorical | Hotel type (City/Resort) |
| `lead_time` | Numeric | Days between booking and arrival |
| `arrival_date_month` | Categorical | Month of arrival |
| `stays_in_weekend_nights` | Numeric | Weekend nights booked |
| `stays_in_week_nights` | Numeric | Weekday nights booked |
| `adults`, `children`, `babies` | Numeric | Guest composition |
| `meal` | Categorical | Meal plan (BB, HB, FB, SC) |
| `country` | Categorical | Guest country of origin |
| `market_segment` | Categorical | Booking channel segment |
| `deposit_type` | Categorical | No Deposit / Non Refund / Refundable |
| `adr` | Numeric | Average Daily Rate (price per night) |
| `total_of_special_requests` | Numeric | Number of special requests |
| `previous_cancellations` | Numeric | Historical cancellation count |
| `is_canceled` | Binary | **TARGET** — 1 if cancelled |

### 2.4 Target Distribution

- Not Cancelled (0): 63.2% (55,212 bookings)
- Cancelled (1): 36.8% (32,018 bookings)
- The dataset is moderately imbalanced. We use stratified splitting and evaluate F1-score (which accounts for imbalance) alongside accuracy.

### 2.5 Ethical & Privacy Considerations

- The dataset is fully anonymised — no names, contact details, or personal identifiers are present
- Country codes are included for demographic analysis only and are not used for discriminatory purposes
- Released under CC0 Public Domain license — no ethical concerns for academic use
- Our model should not be deployed to make decisions that discriminate based on country of origin

---

## 3. Methodology

### 3.1 Data Pre-processing

#### 3.1.1 Removing Duplicates
31,994 exact duplicate rows were identified and removed, leaving 87,396 rows.

#### 3.1.2 Handling Missing Values

| Column | Missing | Strategy |
|---|---|---|
| `children` | 4 rows | Filled with 0 (no children) |
| `country` | 488 rows | Filled with "Unknown" |
| `agent` | 16,340 rows | Filled with 0 (no agent) |
| `company` | 112,593 rows (94%) | Dropped column entirely |

The `company` column was dropped because 94% of its values were null, providing negligible signal.

#### 3.1.3 Removing Logical Inconsistencies
166 bookings with `adults = 0`, `children = 0`, and `babies = 0` were removed as impossible bookings.

#### 3.1.4 Fixing Categorical "Undefined" Values
- `meal`: "Undefined" replaced with "SC" (both represent no meal package — per dataset authors)
- `market_segment`: 2 "Undefined" rows replaced with mode "Online TA"
- `distribution_channel`: 5 "Undefined" rows replaced with mode "TA/TO"

#### 3.1.5 ADR Outlier Handling
Using the IQR method (Q1 - 1.5×IQR to Q3 + 1.5×IQR), 2,508 outliers in the `adr` column were identified and clipped to the bounds. This includes 10 negative values (data entry errors) and extreme values above €5,000.

#### 3.1.6 Leakage Removal
The columns `reservation_status` and `reservation_status_date` were dropped. These are direct proxies for the target (they contain values like "Canceled") and would constitute data leakage if included.

### 3.2 Feature Engineering

Nine new features were derived to enhance model signal:

| Feature | Formula | Rationale |
|---|---|---|
| `total_nights` | `weekend_nights + week_nights` | Total stay duration |
| `total_guests` | `adults + children + babies` | Party size |
| `room_upgraded` | `reserved_type ≠ assigned_type` | Hotel flexibility signal |
| `has_children` | `(children + babies) > 0` | Family booking indicator |
| `revenue_per_night` | `adr × total_nights` | Total booking value |
| `is_long_stay` | `total_nights > 7` | Long-stay indicator |
| `month_num` | Ordinal month (1–12) | Seasonal numeric signal |
| `season` | Winter/Spring/Summer/Autumn | Coarser seasonal signal |
| `cancel_rate_history` | `prev_cancellations / (prev_cancellations + prev_no_cancel + 1)` | Historical cancellation rate |

### 3.3 Preprocessing Pipeline

A scikit-learn `ColumnTransformer` pipeline was built:
- **Numeric features (25):** Median imputation → StandardScaler
- **Categorical features (10):** Mode imputation → OrdinalEncoder (handles unknown categories)

This pipeline is fully serialised with the model to prevent data leakage during prediction.

### 3.4 Train/Test Split

- **Split ratio:** 80% train / 20% test
- **Stratification:** Applied on `is_canceled` to preserve class balance
- **Train set:** 69,784 samples
- **Test set:** 17,446 samples

### 3.5 Algorithms

Five algorithms from distinct model families were trained:

| # | Algorithm | Family | Key Hyperparameters |
|---|---|---|---|
| 1 | Logistic Regression | Linear | max_iter=500, C=1.0 |
| 2 | Decision Tree | Tree-based | max_depth=10 |
| 3 | Random Forest | Ensemble (Bagging) | n_estimators=100, max_depth=15 |
| 4 | XGBoost | Ensemble (Boosting) | n_estimators=100, lr=0.1, max_depth=5 |
| 5 | LightGBM | Ensemble (Boosting) | n_estimators=100, lr=0.1, max_depth=5 |

**Cross-Validation:** 5-fold Stratified K-Fold with ROC-AUC scoring.

---

## 4. Results & Discussion

### 4.1 Model Performance (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7844 | 0.6532 | 0.4094 | 0.5021 | 0.8185 |
| Decision Tree | 0.8260 | 0.7253 | 0.6141 | 0.6659 | 0.8852 |
| Random Forest | 0.8438 | 0.8205 | 0.5747 | 0.6763 | 0.9085 |
| XGBoost | 0.8484 | 0.8052 | 0.6243 | 0.7032 | 0.9116 |
| **LightGBM** | **0.8486** | **0.7762** | **0.6406** | **0.7035** | **0.9122** |

### 4.2 Discussion

**Why LightGBM performed best:**
- Gradient boosting iteratively corrects errors made by previous trees, leading to superior pattern capture
- LightGBM's leaf-wise growth strategy enables it to model complex, non-linear interactions more efficiently than XGBoost's depth-wise approach
- The hotel booking dataset contains many non-linear interactions (e.g., lead_time × deposit_type) that tree-based boosting excels at capturing

**Logistic Regression underperformed** because the relationship between features and cancellation is highly non-linear. The model's recall of 40.9% is particularly weak — it misses most actual cancellations, making it unsuitable for production use.

**Decision Tree vs. Random Forest:** The RF significantly improved over a single tree (ROC-AUC: 0.885 → 0.909) due to variance reduction via bagging. However, it falls short of the boosting models in precision.

**Overfitting analysis:**
- Decision Tree: Training accuracy ~95%, test ~82.6% → mild overfitting controlled by max_depth=10
- Random Forest and boosted models: Small train-test gap (~1-2%) → well-calibrated

**Key features driving predictions:**
1. `lead_time` — Most important predictor; long lead times give more opportunity to cancel
2. `total_of_special_requests` — Strong negative correlation with cancellation (engaged guests keep bookings)
3. `adr` — Higher price sensitivity leads to more cancellation consideration
4. `deposit_type` — Non-refundable deposits should deter cancellation, but paradoxically cancel more (likely because non-refundable rates are often booked through OTAs speculatively)
5. `cancel_rate_history` — Historical behaviour is a reliable predictor of future behaviour

### 4.3 Business Implications

- Bookings with lead_time > 90 days, no deposit, and booked via Online TA are highest risk
- Hotels should request confirmation or apply light incentives to high-risk bookings 30 days before arrival
- Special request collection at time of booking could serve as both a service improvement and a retention mechanism

---

## 5. Conclusion & Future Work

### 5.1 Conclusion

This project successfully built a hotel booking cancellation prediction system achieving:
- **ROC-AUC of 91.2%** (LightGBM) — well above the 85% target
- **F1-Score of 70.4%** — solid balance of precision and recall on an imbalanced dataset
- A complete, reproducible ML pipeline from raw data to Streamlit web application

The most important finding is that **lead time and special requests are the dominant predictors**, which aligns with intuitive business understanding. Ensemble boosting methods (LightGBM, XGBoost) are significantly superior to linear and single-tree approaches for this task.

### 5.2 Future Work

1. **Threshold optimisation:** Adjust the decision threshold (currently 0.5) based on business cost of false positives vs false negatives
2. **SMOTE / class-weight balancing:** Address the 63/37 class imbalance more formally
3. **Time-series cross-validation:** Since the data spans 2015–2017, temporal splits would be more realistic than random splits
4. **SHAP explanations:** Implement SHAP values for booking-level explainability (regulatory requirement in some markets)
5. **Real-time API:** Deploy the model as a REST API using FastAPI for integration with hotel PMS systems
6. **Feature addition:** Incorporate external data (weather, events, economic indicators) to improve seasonal predictions

---

## 6. References

1. Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.
3. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NIPS 2017*.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
5. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

---

## 7. Individual Reflection

*(Each team member should complete this 1-page section individually, describing their specific contributions and what they learned from the project.)*

**Member 1 — [Name]:**
> [Role: Data cleaning, EDA, and preprocessing pipeline. What I learned: How to handle real-world data quality issues and the importance of preventing data leakage in ML pipelines...]

**Member 2 — [Name]:**
> [Role: Model implementation, cross-validation, and hyperparameter tuning. What I learned: The dramatic difference between linear and ensemble methods on non-linear problems, and how to properly validate models...]

**Member 3 — [Name]:**
> [Role: Evaluation, visualisations, Streamlit app, and report writing. What I learned: The importance of choosing the right evaluation metric for imbalanced classification, and how to translate model outputs into business insights...]
