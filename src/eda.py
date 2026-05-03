"""
eda.py
======
Exploratory Data Analysis for the Hotel Booking Cancellation project.
Generates and saves all required figures to outputs/figures/.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

# ── Palette ─────────────────────────────────────────────────────────────────
BRIGHT_BLUE  = "#007BFF"
ACCENT_RED   = "#E63946"
ACCENT_GREEN = "#2DC653"
PALETTE_CAT  = [BRIGHT_BLUE, ACCENT_RED, ACCENT_GREEN, "#FFC300", "#9B5DE5", "#00B4D8"]

FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams["figure.dpi"] = 150

MONTH_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December",
]


def _save(fig: plt.Figure, name: str):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [eda] Saved → {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. TARGET DISTRIBUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_target_distribution(df: pd.DataFrame):
    total       = len(df)
    cancelled   = df["is_canceled"].sum()
    kept        = total - cancelled

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("KPI 1 & 2 — Cancellation Overview", fontsize=14, fontweight="bold")

    # Pie
    axes[0].pie(
        [cancelled, kept],
        labels=["Cancelled", "Not Cancelled"],
        colors=[ACCENT_RED, BRIGHT_BLUE],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    axes[0].set_title("Overall Cancellation Rate")

    # Bar by hotel type
    cancel_hotel = (
        df.groupby("hotel")["is_canceled"]
          .agg(cancelled="sum", total="count")
          .assign(rate=lambda x: x["cancelled"] / x["total"] * 100)
    )
    bars = axes[1].bar(cancel_hotel.index, cancel_hotel["rate"],
                       color=[BRIGHT_BLUE, ACCENT_RED], edgecolor="white")
    for bar, v in zip(bars, cancel_hotel["rate"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, f"{v:.1f}%", ha="center", fontsize=11)
    axes[1].set_title("Cancellation Rate by Hotel Type")
    axes[1].set_ylabel("Cancellation Rate (%)")
    axes[1].set_ylim(0, 60)

    _save(fig, "01_target_distribution")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. MONTHLY BOOKINGS & CANCELLATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_monthly_trends(df: pd.DataFrame):
    df = df.copy()
    df["arrival_date_month"] = pd.Categorical(
        df["arrival_date_month"], categories=MONTH_ORDER, ordered=True
    )
    monthly = (
        df.groupby("arrival_date_month", observed=True)["is_canceled"]
          .agg(total="count", cancelled="sum")
          .assign(kept=lambda x: x["total"] - x["cancelled"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("KPI 3 — Monthly Booking & Cancellation Trends", fontsize=14, fontweight="bold")

    x = range(len(monthly))
    w = 0.4
    axes[0].bar([i - w/2 for i in x], monthly["kept"],      width=w, label="Confirmed", color=BRIGHT_BLUE)
    axes[0].bar([i + w/2 for i in x], monthly["cancelled"], width=w, label="Cancelled", color=ACCENT_RED)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([m[:3] for m in MONTH_ORDER], rotation=45)
    axes[0].set_title("Bookings by Month")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    cancel_rate = monthly["cancelled"] / monthly["total"] * 100
    axes[1].plot(list(x), cancel_rate, marker="o", color=ACCENT_RED, linewidth=2)
    axes[1].fill_between(list(x), cancel_rate, alpha=0.15, color=ACCENT_RED)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([m[:3] for m in MONTH_ORDER], rotation=45)
    axes[1].set_title("Monthly Cancellation Rate (%)")
    axes[1].set_ylabel("%")
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

    _save(fig, "02_monthly_trends")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. LEAD TIME DISTRIBUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_lead_time(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("KPI 4 — Lead Time vs Cancellation", fontsize=14, fontweight="bold")

    for label, color, mask in [
        ("Not Cancelled", BRIGHT_BLUE, df["is_canceled"] == 0),
        ("Cancelled",     ACCENT_RED,  df["is_canceled"] == 1),
    ]:
        axes[0].hist(df.loc[mask, "lead_time"], bins=50, alpha=0.6, label=label, color=color)
    axes[0].set_title("Lead Time Distribution by Cancellation")
    axes[0].set_xlabel("Lead Time (days)")
    axes[0].legend()

    bins = [0, 7, 30, 90, 180, 365, 9999]
    labels_b = ["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", ">365d"]
    df2 = df.copy()
    df2["lead_bin"] = pd.cut(df2["lead_time"], bins=bins, labels=labels_b)
    grp = df2.groupby("lead_bin", observed=True)["is_canceled"].mean() * 100
    axes[1].bar(grp.index, grp.values, color=PALETTE_CAT[:len(grp)])
    axes[1].set_title("Cancellation Rate by Lead Time Bucket")
    axes[1].set_ylabel("Cancellation Rate (%)")
    axes[1].set_xlabel("Lead Time Bucket")
    for i, v in enumerate(grp.values):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    _save(fig, "03_lead_time")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. DEPOSIT TYPE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_deposit_type(df: pd.DataFrame):
    grp = df.groupby("deposit_type")["is_canceled"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(grp.index, grp.values, color=PALETTE_CAT[:len(grp)], edgecolor="white")
    for bar, v in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_title("KPI 5 — Cancellation Rate by Deposit Type", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cancellation Rate (%)")
    ax.set_xlabel("Deposit Type")
    _save(fig, "04_deposit_type")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. MARKET SEGMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_market_segment(df: pd.DataFrame):
    grp = (
        df.groupby("market_segment")["is_canceled"]
          .agg(cancel_rate="mean", total="count")
          .sort_values("cancel_rate", ascending=False)
    )
    grp["cancel_rate"] *= 100

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(grp.index, grp["cancel_rate"], color=PALETTE_CAT[:len(grp)], edgecolor="white")
    for bar, v in zip(bars, grp["cancel_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_title("KPI 6 — Cancellation Rate by Market Segment", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cancellation Rate (%)")
    ax.set_xlabel("Market Segment")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, "05_market_segment")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. CORRELATION HEATMAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_correlation_heatmap(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
        linewidths=0.5, ax=ax, annot_kws={"size": 7},
        vmin=-1, vmax=1, center=0,
    )
    ax.set_title("Numerical Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    _save(fig, "06_correlation_heatmap")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. ADR DISTRIBUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_adr(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("KPI 7 — Average Daily Rate (ADR) Analysis", fontsize=14, fontweight="bold")

    for label, color, mask in [
        ("Not Cancelled", BRIGHT_BLUE, df["is_canceled"] == 0),
        ("Cancelled",     ACCENT_RED,  df["is_canceled"] == 1),
    ]:
        axes[0].hist(df.loc[mask, "adr"], bins=50, alpha=0.6, label=label, color=color)
    axes[0].set_title("ADR Distribution by Cancellation")
    axes[0].set_xlabel("ADR (€)")
    axes[0].legend()

    df.groupby("hotel")["adr"].plot(kind="kde", ax=axes[1])
    axes[1].set_title("ADR Density by Hotel Type")
    axes[1].set_xlabel("ADR (€)")
    axes[1].legend(["City Hotel", "Resort Hotel"])

    _save(fig, "07_adr_distribution")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. SPECIAL REQUESTS vs CANCELLATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_special_requests(df: pd.DataFrame):
    grp = df.groupby("total_of_special_requests")["is_canceled"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grp.index, grp.values, color=BRIGHT_BLUE, edgecolor="white")
    ax.set_title("KPI 8 — Cancellation Rate by Special Requests", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Special Requests")
    ax.set_ylabel("Cancellation Rate (%)")
    _save(fig, "08_special_requests")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. CUSTOMER TYPE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_customer_type(df: pd.DataFrame):
    grp = df.groupby("customer_type")["is_canceled"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(grp.index, grp.values, color=PALETTE_CAT[:len(grp)], edgecolor="white")
    for bar, v in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_title("KPI 9 — Cancellation Rate by Customer Type", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cancellation Rate (%)")
    ax.set_xlabel("Customer Type")
    _save(fig, "09_customer_type")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. TOP COUNTRIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_top_countries(df: pd.DataFrame):
    top = df["country"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top.index[::-1], top.values[::-1], color=BRIGHT_BLUE, edgecolor="white")
    ax.set_title("KPI 10 — Top 15 Guest Countries", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Bookings")
    _save(fig, "10_top_countries")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUN ALL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_all_eda(df: pd.DataFrame):
    print("[eda] Generating all EDA figures …")
    plot_target_distribution(df)
    plot_monthly_trends(df)
    plot_lead_time(df)
    plot_deposit_type(df)
    plot_market_segment(df)
    plot_correlation_heatmap(df)
    plot_adr(df)
    plot_special_requests(df)
    plot_customer_type(df)
    plot_top_countries(df)
    print(f"[eda] Done. All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_raw
    from preprocessing import clean, engineer_features
    df = engineer_features(clean(load_raw()))
    run_all_eda(df)
