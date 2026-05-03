"""
pipeline.py
===========
Master script: runs the full ML pipeline end-to-end.

  python pipeline.py [--skip-eda] [--skip-eval]

Steps
-----
1. Load raw data
2. Clean data
3. Feature engineering
4. EDA (visualizations)
5. Train & tune models
6. Evaluate & compare models
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader    import load_raw, basic_report
from preprocessing  import clean, engineer_features, prepare_data
from eda            import run_all_eda
from train          import run_training
from evaluate       import run_evaluation


def main(skip_eda: bool = False, skip_eval: bool = False):
    print("\n" + "━"*60)
    print("  HOTEL BOOKING CANCELLATION PREDICTION")
    print("  Full ML Pipeline")
    print("━"*60)

    # ── Step 1: Load ────────────────────────────────────────────────────────
    print("\n[pipeline] STEP 1 — Load Data")
    raw = load_raw()
    basic_report(raw)

    # ── Step 2: Clean ───────────────────────────────────────────────────────
    print("\n[pipeline] STEP 2 — Data Cleaning")
    clean_df = clean(raw)

    # ── Step 3: Feature Engineering ─────────────────────────────────────────
    print("\n[pipeline] STEP 3 — Feature Engineering")
    feat_df = engineer_features(clean_df)

    # ── Step 4: EDA ─────────────────────────────────────────────────────────
    if not skip_eda:
        print("\n[pipeline] STEP 4 — EDA & Visualizations")
        run_all_eda(feat_df)
    else:
        print("\n[pipeline] STEP 4 — EDA skipped (--skip-eda)")

    # ── Step 5: Train ───────────────────────────────────────────────────────
    print("\n[pipeline] STEP 5 — Training & Hyperparameter Tuning")
    best_model, results_df = run_training()

    # ── Step 6: Evaluate ────────────────────────────────────────────────────
    if not skip_eval:
        print("\n[pipeline] STEP 6 — Evaluation & Model Comparison")
        run_evaluation()
    else:
        print("\n[pipeline] STEP 6 — Evaluation skipped (--skip-eval)")

    print("\n" + "━"*60)
    print("  PIPELINE COMPLETE")
    print("━"*60)
    print("  Best model  → models/best_model.pkl")
    print("  Results CSV → models/model_results.csv")
    print("  Figures     → outputs/figures/")
    print("  Run Streamlit app: streamlit run app.py")
    print("━"*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hotel Booking ML Pipeline")
    parser.add_argument("--skip-eda",  action="store_true", help="Skip EDA plots")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation plots")
    args = parser.parse_args()
    main(skip_eda=args.skip_eda, skip_eval=args.skip_eval)
