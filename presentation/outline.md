# 🏨 Hotel Booking Cancellation Prediction
## EED472 — 10-Minute Presentation Outline

---

### SLIDE 1 — Title Slide (30 sec)

**Title:** Hotel Booking Cancellation Prediction  
**Subtitle:** EED472 Machine Learning Project — Spring 2026  
**Team:** [Member 1], [Member 2], [Member 3]  
**University:** Future University in Egypt

*Visual: Hotel background image + project logo*

---

### SLIDE 2 — Problem Statement (45 sec)

**The Problem:**
- 30–40% of hotel bookings are cancelled globally
- Last-minute cancellations cause revenue loss and operational chaos
- Hotels need to predict cancellations *at booking time*

**Our Solution:**
- Binary classification model: Will this booking be cancelled?
- Target: `is_canceled` (1 = cancelled, 0 = kept)
- Enables: overbooking strategy, targeted retention, revenue optimisation

*Visual: Simple diagram — "Booking Made" → [ML Model] → "Keep / Cancel Prediction"*

---

### SLIDE 3 — Dataset Overview (45 sec)

**Dataset:** Hotel Booking Demand (Kaggle)

| Stat | Value |
|---|---|
| Original size | 119,390 bookings |
| After cleaning | 87,230 bookings |
| Features | 32 original → 35 engineered |
| Hotels | City Hotel (66%) + Resort Hotel (34%) |
| Cancellation rate | 36.8% |

**Key features:** Lead time, deposit type, market segment, ADR, special requests

*Visual: Pie chart — 63% not cancelled / 37% cancelled*

---

### SLIDE 4 — Data Pipeline (60 sec)

**Step-by-step pipeline:**

```
Raw CSV → Clean → Engineer Features → EDA → Preprocess → Train → Evaluate → Deploy
```

**Cleaning highlights:**
- Removed 31,994 duplicates
- Filled nulls (agent, children, country)
- Removed 166 zero-guest bookings  
- Clipped ADR outliers (IQR method)
- ⚠️ Dropped `reservation_status` (data leakage!)

**New features engineered:**
- `total_nights`, `total_guests`, `room_upgraded`
- `cancel_rate_history`, `season`, `has_children`

*Visual: Pipeline flowchart with color-coded steps*

---

### SLIDE 5 — EDA Key Findings (90 sec)

**KPI 1:** City Hotel cancels at 41.7% vs Resort Hotel at 27.8%

**KPI 2:** Lead time drives cancellation — bookings made >6 months out cancel 55%+ of the time

**KPI 3:** Special requests = commitment — 0 requests → 43% cancel; 5 requests → <5% cancel

**KPI 4:** Deposit type is key:
- No Deposit: ~36% cancel
- Non Refundable: ~99% cancel (speculative bookings!)
- Refundable: ~22% cancel

**KPI 5:** Online TA segment has the highest cancellation rate

*Visual: 2×2 grid of the most impactful charts (deposit type, lead time, special requests, monthly)*

---

### SLIDE 6 — Models Trained (45 sec)

**Five algorithms from different families:**

| # | Model | Family |
|---|---|---|
| 1 | Logistic Regression | Linear |
| 2 | Decision Tree | Single Tree |
| 3 | Random Forest | Ensemble — Bagging |
| 4 | XGBoost | Ensemble — Boosting |
| 5 | **LightGBM** ✅ | **Ensemble — Boosting** |

**Validation:** 5-fold Stratified K-Fold Cross-Validation  
**Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC

---

### SLIDE 7 — Results & Comparison (90 sec)

**Test Set Performance:**

| Model | ROC-AUC | F1 |
|---|---|---|
| Logistic Regression | 0.819 | 0.502 |
| Decision Tree | 0.885 | 0.666 |
| Random Forest | 0.909 | 0.676 |
| XGBoost | 0.912 | 0.703 |
| **LightGBM** | **0.912** | **0.704** |

**LightGBM wins** with ROC-AUC = 91.2% and F1 = 70.4%

*Visual: Grouped bar chart of all 5 metrics across all 5 models + ROC curve overlay*

---

### SLIDE 8 — Feature Importance (45 sec)

**Top 5 Most Important Features (LightGBM):**

1. 🥇 `lead_time` — Long booking horizon → higher risk
2. 🥈 `total_of_special_requests` — More engaged = less likely to cancel
3. 🥉 `adr` — Price sensitivity
4. `deposit_type` — Non-refundable behaves counter-intuitively
5. `cancel_rate_history` — Past behaviour predicts future

*Visual: Horizontal bar chart of top 20 feature importances*

---

### SLIDE 9 — Streamlit App Demo (60 sec)

**Live Demo of the Web Application:**

📊 **Dashboard Tab:** Interactive EDA — monthly trends, market segments, KPI cards

🔮 **Predictor Tab:**
- Enter booking details (lead time, deposit type, hotel type, etc.)
- Get cancellation probability + gauge chart
- Example: 200-day lead time + No Deposit + Online TA → 78% cancellation risk

📈 **Model Results Tab:** Performance table + radar chart + feature importance

*Visual: Screenshot of the Streamlit app (all 3 tabs)*

---

### SLIDE 10 — Conclusion & Future Work (45 sec)

**Achievements:**
- ✅ Built a full ML pipeline (data → model → web app)
- ✅ LightGBM achieved ROC-AUC of 91.2% (target was ≥ 85%)
- ✅ Identified lead time, special requests, and deposit type as key drivers

**Future Work:**
- SHAP explainability for individual booking decisions
- SMOTE to formally address class imbalance
- REST API deployment for hotel PMS integration
- Incorporate external data (weather, local events)

**Business Impact:**
> A hotel using this model could save €15,000–50,000/year by reducing excess overbooking costs and targeting retention efforts on high-risk bookings

---

### Q&A Slide (30 sec)

**Thank You!**

> *"Data is not just oil — it's the intelligence that transforms raw bookings into business strategy."*

**Questions?**

GitHub: [your-repo-link]  
Contact: [team-email]

---

## Presentation Tips

- **Timing guide:** Slides 1–5: Introduction + EDA (~4 min) | Slides 6–8: Models (~3 min) | Slides 9–10: App + Conclusion (~3 min)
- **All team members must speak** — divide sections (e.g., Member 1: Problem + Data, Member 2: Models, Member 3: Results + App)
- **Bring a laptop** for the live Streamlit demo
- **Key numbers to memorise:** 119,390 bookings, 87,230 after cleaning, 36.8% cancellation rate, LightGBM AUC=91.2%
- **Anticipate this question:** "Why is Non-Refundable deposit type associated with more cancellations?" — Answer: These are often speculative bookings made through OTAs where guests are rate-shopping
