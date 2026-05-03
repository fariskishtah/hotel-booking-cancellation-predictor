"""
app.py
======
Streamlit application for the Hotel Booking Cancellation Prediction project.

Tabs
----
1. 📊 Dashboard  — EDA visualisations and KPI summary
2. 🔮 Predictor  — Enter a new booking and get a cancellation prediction
3. 📈 Model Results — Performance metrics and comparison charts
4. ℹ️  About      — Project info

Run:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
ASSET_DIR = BASE_DIR / "assets"
FIG_DIR   = BASE_DIR / "outputs" / "figures"
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "hotel_bookings.csv"

BRIGHT_BLUE  = "#007BFF"
ACCENT_RED   = "#E63946"
ACCENT_GREEN = "#2DC653"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #007BFF22, #007BFF11);
        border-left: 4px solid #007BFF;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 10px;
    }
    .cancel-card {
        background: linear-gradient(135deg, #E6394622, #E6394611);
        border-left: 4px solid #E63946;
        border-radius: 8px;
        padding: 12px 18px;
    }
    .green-card {
        background: linear-gradient(135deg, #2DC65322, #2DC65311);
        border-left: 4px solid #2DC653;
        border-radius: 8px;
        padding: 12px 18px;
    }
    .stTabs [data-baseweb='tab'] { font-size: 16px; }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CACHED LOADERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data
def load_data():
    from data_loader import load_raw
    from preprocessing import clean, engineer_features
    raw = load_raw()
    return engineer_features(clean(raw))

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "best_model.pkl")

@st.cache_data
def load_results():
    return pd.read_csv(MODEL_DIR / "model_results.csv")

@st.cache_data
def load_feature_importance():
    path = MODEL_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, names=["feature","importance"]).dropna()
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.image(str(ASSET_DIR / "logo.png"), width=150)
    st.title("🏨 Hotel Booking\nCancellation\nPredictor")
    st.markdown("---")
    st.markdown("**EED472 — Machine Learning Project**")
    st.markdown("Future University in Egypt · Spring 2026")
    st.markdown("---")
    st.markdown("### Dataset Stats")
    try:
        df = load_data()
        st.metric("Total Bookings", f"{len(df):,}")
        st.metric("Cancellation Rate", f"{df['is_canceled'].mean()*100:.1f}%")
        st.metric("Features Used", "35")
    except Exception:
        st.info("Load data first")
    st.markdown("---")
    st.markdown("**Best Model:** LightGBM")
    st.markdown("**ROC-AUC:** 0.9122")
    st.markdown("**F1 Score:** 0.7035")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "🔮 Predictor", "📈 Model Results", "ℹ️ About"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.title("📊 Hotel Booking EDA Dashboard")
    st.markdown("Interactive exploration of the hotel bookings dataset.")

    try:
        df = load_data()

        # KPI row
        total = len(df)
        cancelled = df["is_canceled"].sum()
        avg_adr = df[df["is_canceled"] == 0]["adr"].mean()
        avg_lead = df["lead_time"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Bookings", f"{total:,}")
        c2.metric("Cancellations", f"{cancelled:,}", f"{cancelled/total*100:.1f}%")
        c3.metric("Avg ADR (Confirmed)", f"€{avg_adr:.0f}")
        c4.metric("Avg Lead Time", f"{avg_lead:.0f} days")

        st.markdown("---")

        # Plot 1: Cancellation by hotel
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cancellation by Hotel Type")
            hotel_grp = (
                df.groupby("hotel")["is_canceled"]
                  .agg(cancelled="sum", total="count")
                  .assign(rate=lambda x: x["cancelled"]/x["total"]*100)
                  .reset_index()
            )
            fig = px.bar(
                hotel_grp, x="hotel", y="rate",
                color="hotel", color_discrete_sequence=[BRIGHT_BLUE, ACCENT_RED],
                text=hotel_grp["rate"].apply(lambda v: f"{v:.1f}%"),
                labels={"rate": "Cancellation Rate (%)", "hotel": "Hotel Type"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Cancellation by Deposit Type")
            dep_grp = (
                df.groupby("deposit_type")["is_canceled"]
                  .mean().mul(100).reset_index()
                  .rename(columns={"is_canceled": "cancel_rate"})
                  .sort_values("cancel_rate", ascending=False)
            )
            fig = px.bar(
                dep_grp, x="deposit_type", y="cancel_rate",
                color="cancel_rate", color_continuous_scale=["#007BFF","#E63946"],
                text=dep_grp["cancel_rate"].apply(lambda v: f"{v:.1f}%"),
                labels={"cancel_rate": "Cancellation Rate (%)", "deposit_type": "Deposit Type"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Plot 2: Monthly trends
        st.subheader("Monthly Booking Trends")
        MONTH_ORDER = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        df2 = df.copy()
        df2["arrival_date_month"] = pd.Categorical(
            df2["arrival_date_month"], categories=MONTH_ORDER, ordered=True
        )
        monthly = (
            df2.groupby("arrival_date_month", observed=True)["is_canceled"]
               .agg(total="count", cancelled="sum")
               .assign(confirmed=lambda x: x["total"] - x["cancelled"],
                       cancel_rate=lambda x: x["cancelled"]/x["total"]*100)
               .reset_index()
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(name="Confirmed", x=monthly["arrival_date_month"],
                             y=monthly["confirmed"], marker_color=BRIGHT_BLUE), secondary_y=False)
        fig.add_trace(go.Bar(name="Cancelled", x=monthly["arrival_date_month"],
                             y=monthly["cancelled"], marker_color=ACCENT_RED), secondary_y=False)
        fig.add_trace(go.Scatter(name="Cancel Rate %", x=monthly["arrival_date_month"],
                                 y=monthly["cancel_rate"], mode="lines+markers",
                                 line=dict(color="#FFC300", width=2.5)), secondary_y=True)
        fig.update_layout(barmode="stack", height=380, legend=dict(orientation="h"))
        fig.update_yaxes(title_text="Bookings", secondary_y=False)
        fig.update_yaxes(title_text="Cancel Rate (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        # Plot 3: Lead Time
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Lead Time Distribution")
            fig = px.histogram(
                df, x="lead_time", color=df["is_canceled"].map({0:"Not Cancelled",1:"Cancelled"}),
                nbins=60, barmode="overlay", opacity=0.7,
                color_discrete_map={"Not Cancelled": BRIGHT_BLUE, "Cancelled": ACCENT_RED},
                labels={"lead_time": "Lead Time (days)", "color": "Status"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("Market Segment Analysis")
            seg_grp = (
                df.groupby("market_segment")["is_canceled"]
                  .agg(cancel_rate="mean", count="count")
                  .assign(cancel_rate=lambda x: x["cancel_rate"]*100)
                  .sort_values("cancel_rate", ascending=False)
                  .reset_index()
            )
            fig = px.scatter(
                seg_grp, x="count", y="cancel_rate", size="count",
                text="market_segment", color="cancel_rate",
                color_continuous_scale=["#007BFF","#E63946"],
                labels={"count": "Total Bookings", "cancel_rate": "Cancellation Rate (%)"},
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Plot 4: Special requests
        st.subheader("Special Requests vs Cancellation Rate")
        sr_grp = (
            df.groupby("total_of_special_requests")["is_canceled"]
              .mean().mul(100).reset_index()
              .rename(columns={"is_canceled": "cancel_rate"})
        )
        fig = px.bar(
            sr_grp, x="total_of_special_requests", y="cancel_rate",
            color="cancel_rate", color_continuous_scale=["#2DC653","#007BFF","#E63946"],
            labels={"total_of_special_requests": "Number of Special Requests",
                    "cancel_rate": "Cancellation Rate (%)"},
            text=sr_grp["cancel_rate"].apply(lambda v: f"{v:.1f}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load dashboard: {e}")
        st.info("Run `python src/pipeline.py` first to generate the model and data.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.title("🔮 Booking Cancellation Predictor")
    st.markdown("Enter booking details below to predict the probability of cancellation.")

    try:
        model = load_model()

        with st.form("prediction_form"):
            st.subheader("Booking Details")
            col1, col2, col3 = st.columns(3)

            with col1:
                hotel         = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
                lead_time     = st.number_input("Lead Time (days)", 0, 730, 60)
                deposit_type  = st.selectbox("Deposit Type",
                                             ["No Deposit", "Non Refund", "Refundable"])
                market_seg    = st.selectbox("Market Segment",
                                             ["Online TA","Offline TA/TO","Direct",
                                              "Corporate","Groups","Aviation","Complementary"])
                dist_channel  = st.selectbox("Distribution Channel",
                                             ["TA/TO","Direct","Corporate","GDS","Undefined"])

            with col2:
                adults        = st.number_input("Adults", 1, 10, 2)
                children      = st.number_input("Children", 0, 10, 0)
                babies        = st.number_input("Babies", 0, 5, 0)
                weekend_nights= st.number_input("Weekend Nights", 0, 15, 1)
                week_nights   = st.number_input("Week Nights", 0, 30, 2)

            with col3:
                adr           = st.number_input("ADR (€)", 0.0, 600.0, 100.0, step=5.0)
                special_req   = st.slider("Special Requests", 0, 5, 0)
                meal          = st.selectbox("Meal Plan", ["BB","FB","HB","SC","Undefined"])
                customer_type = st.selectbox("Customer Type",
                                             ["Transient","Contract","Transient-Party","Group"])
                prev_cancel   = st.number_input("Previous Cancellations", 0, 20, 0)
                prev_no_cancel= st.number_input("Previous Non-Cancellations", 0, 50, 0)

            submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

        if submitted:
            total_nights   = weekend_nights + week_nights
            total_guests   = adults + children + babies
            room_upgraded  = 0
            has_children   = int((children + babies) > 0)
            revenue        = adr * max(total_nights, 1)
            is_long_stay   = int(total_nights > 7)
            month_num      = 6
            season         = "Summer"
            cancel_hist    = prev_cancel / (prev_cancel + prev_no_cancel + 1)

            input_data = {
                "lead_time": lead_time,
                "arrival_date_year": 2025,
                "arrival_date_day_of_month": 15,
                "stays_in_weekend_nights": weekend_nights,
                "stays_in_week_nights": week_nights,
                "adults": adults,
                "children": children,
                "babies": babies,
                "is_repeated_guest": 0,
                "previous_cancellations": prev_cancel,
                "previous_bookings_not_canceled": prev_no_cancel,
                "booking_changes": 0,
                "agent": 9,
                "days_in_waiting_list": 0,
                "adr": adr,
                "required_car_parking_spaces": 0,
                "total_of_special_requests": special_req,
                "total_nights": total_nights,
                "total_guests": total_guests,
                "room_upgraded": room_upgraded,
                "has_children": has_children,
                "revenue_per_night": revenue,
                "is_long_stay": is_long_stay,
                "month_num": month_num,
                "cancel_rate_history": cancel_hist,
                "hotel": hotel,
                "meal": meal,
                "country": "PRT",
                "market_segment": market_seg,
                "distribution_channel": dist_channel,
                "reserved_room_type": "A",
                "assigned_room_type": "A",
                "deposit_type": deposit_type,
                "customer_type": customer_type,
                "season": season,
            }

            X_input = pd.DataFrame([input_data])
            prob = model.predict_proba(X_input)[0, 1]
            pred = int(prob >= 0.5)

            st.markdown("---")
            st.subheader("Prediction Result")
            col_res1, col_res2, col_res3 = st.columns(3)

            with col_res1:
                if pred == 1:
                    st.error(f"⚠️ **HIGH RISK OF CANCELLATION**")
                else:
                    st.success(f"✅ **LIKELY TO KEEP BOOKING**")

            with col_res2:
                st.metric("Cancellation Probability", f"{prob*100:.1f}%")

            with col_res3:
                st.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                title={"text": "Cancellation Risk (%)"},
                delta={"reference": 37, "suffix": "% vs avg"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": ACCENT_RED if prob > 0.5 else BRIGHT_BLUE},
                    "steps": [
                        {"range": [0, 30],  "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 100],"color": "#f8d7da"},
                    ],
                    "threshold": {"line": {"color": "black","width": 3},"value": 50},
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔍 Input Summary"):
                st.json(input_data)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("Make sure the model is trained. Run `python src/pipeline.py` first.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.title("📈 Model Performance & Comparison")

    try:
        results = load_results()

        # Summary table
        st.subheader("Performance Summary Table")
        styled = results[["model","accuracy","precision","recall","f1","roc_auc"]]\
                    .sort_values("roc_auc", ascending=False)\
                    .reset_index(drop=True)
        st.dataframe(
            styled.style
                  .background_gradient(subset=["accuracy","precision","recall","f1","roc_auc"],
                                       cmap="RdYlGn", vmin=0.5, vmax=1.0)
                  .format({c: "{:.4f}" for c in ["accuracy","precision","recall","f1","roc_auc"]}),
            use_container_width=True,
        )

        # Radar chart
        st.subheader("Model Comparison — Radar Chart")
        metrics = ["accuracy","precision","recall","f1","roc_auc"]
        COLORS = [BRIGHT_BLUE, ACCENT_RED, "#2DC653", "#FFC300", "#9B5DE5"]
        fig = go.Figure()
        for (_, row), color in zip(results.iterrows(), COLORS):
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics] + [row[metrics[0]]],
                theta=metrics + [metrics[0]],
                fill="toself", name=row["model"],
                line_color=color, fillcolor=color,
                opacity=0.25,
            ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0.5, 1.0])),
                          height=480, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart per metric
        st.subheader("Metric-by-Metric Breakdown")
        melted = results[["model"] + metrics].melt(id_vars="model", var_name="Metric", value_name="Score")
        fig = px.bar(
            melted, x="model", y="Score", color="Metric",
            barmode="group", range_y=[0.4, 1.0],
            color_discrete_sequence=COLORS,
        )
        fig.update_layout(height=420, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        fi = load_feature_importance()
        if fi is not None:
            st.subheader("Top 20 Feature Importances (LightGBM)")
            fi_top = fi.head(20).sort_values("importance")
            fig = px.bar(
                fi_top, x="importance", y=fi_top.index,
                orientation="h", color="importance",
                color_continuous_scale=["#00B4D8", BRIGHT_BLUE],
                labels={"importance": "Importance Score", "y": "Feature"},
            )
            fig.update_layout(height=550, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Show saved figures
        st.subheader("Saved Evaluation Figures")
        fig_files = sorted(FIG_DIR.glob("1[1-5]_*.png"))
        if fig_files:
            for img_path in fig_files:
                st.image(str(img_path), caption=img_path.stem.replace("_"," ").title(),
                         use_column_width=True)
        else:
            st.info("Run the pipeline to generate evaluation figures.")

    except Exception as e:
        st.error(f"Could not load results: {e}")
        st.info("Run `python src/pipeline.py` first.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## Hotel Booking Cancellation Prediction

    **Course:** EED472 — Machine Learning and Pattern Recognition  
    **University:** Future University in Egypt  
    **Semester:** Spring 2026  
    **Team Size:** 3 students

    ---

    ### 🎯 Project Objective
    Predict whether a hotel booking will be cancelled before the arrival date,
    enabling hotels to optimise overbooking strategies and revenue management.

    ### 📊 Dataset
    - **Source:** Kaggle — Hotel Booking Demand dataset
    - **Original size:** 119,390 bookings × 32 features
    - **After cleaning:** 87,230 bookings × 35 features (after feature engineering)
    - **Target variable:** `is_canceled` (binary: 0 = kept, 1 = cancelled)
    - **Class balance:** ~63% not cancelled / 37% cancelled

    ### 🔧 Pipeline
    1. **Data Cleaning** — nulls, duplicates, zero-guest rows, undefined categoricals, ADR outliers
    2. **Feature Engineering** — 10 derived features (total_nights, cancel_rate_history, season, etc.)
    3. **EDA** — 10 KPI visualisations covering cancellation rates, lead time, ADR, countries
    4. **Preprocessing** — StandardScaler (numeric) + OrdinalEncoder (categorical)
    5. **Modelling** — 5 algorithms from different families
    6. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC + comparison charts

    ### 🤖 Models Trained
    | Model | Family |
    |---|---|
    | Logistic Regression | Linear |
    | Decision Tree | Tree-based |
    | Random Forest | Ensemble (Bagging) |
    | XGBoost | Ensemble (Boosting) |
    | LightGBM ✅ (Best) | Ensemble (Boosting) |

    ### 🏆 Best Model — LightGBM
    | Metric | Score |
    |---|---|
    | Accuracy | 84.86% |
    | Precision | 77.62% |
    | Recall | 64.06% |
    | F1 Score | 70.35% |
    | ROC-AUC | **91.22%** |

    ### 🔑 Key Findings
    - **Lead time** is the strongest predictor — longer lead times → higher cancellation risk
    - **Deposit type** matters greatly — "Non Refund" bookings cancel at nearly 100%
    - **Special requests** reduce cancellation probability significantly
    - **Online TA** market segment has the highest cancellation rate
    - LightGBM and XGBoost significantly outperform linear models

    ### 📁 Project Structure
    ```
    hotel_ml_project/
    ├── src/
    │   ├── data_loader.py        # Data loading
    │   ├── preprocessing.py      # Cleaning + feature engineering + sklearn Pipeline
    │   ├── eda.py                # All EDA visualisations
    │   ├── train.py              # Model training + hyperparameter tuning
    │   ├── evaluate.py           # Evaluation + comparison plots
    │   └── pipeline.py           # End-to-end master script
    ├── data/hotel_bookings.csv
    ├── models/                   # Saved model + results
    ├── outputs/figures/          # All generated plots
    ├── report/final_report.md
    ├── presentation/outline.md
    ├── app.py                    # This Streamlit app
    ├── requirements.txt
    └── README.md
    ```
    """)
