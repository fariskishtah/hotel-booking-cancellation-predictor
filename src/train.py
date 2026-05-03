"""
train.py
========
Trains five binary classification models for hotel booking cancellation
prediction and saves the best model + results to disk.

Models
------
1. Logistic Regression   (Linear)
2. Decision Tree         (Tree-based)
3. Random Forest         (Ensemble – Bagging)
4. XGBoost               (Ensemble – Boosting)
5. LightGBM              (Ensemble – Boosting)  ← best

Run:  python src/train.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
import xgboost as xgb
import lightgbm as lgb

from data_loader    import load_raw
from preprocessing  import clean, engineer_features, prepare_data

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate(model, X_test, y_test, name: str) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score (y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred),  4),
        "recall":    round(recall_score   (y_test, y_pred),  4),
        "f1":        round(f1_score       (y_test, y_pred),  4),
        "roc_auc":   round(roc_auc_score  (y_test, y_proba), 4),
    }
    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<12}: {v:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Cancelled","Cancelled"]))
    return metrics


def cv_score(pipeline, X_train, y_train, name: str) -> float:
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=CV, scoring="roc_auc", n_jobs=-1)
    mean, std = scores.mean(), scores.std()
    print(f"  [{name}]  CV ROC-AUC: {mean:.4f} +/- {std:.4f}")
    return float(mean)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_base_models(preprocessor) -> dict:
    """Return a fresh dict of un-fitted Pipelines."""
    return {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor),
            ("clf",  LogisticRegression(max_iter=500, C=1.0,
                                        random_state=42, n_jobs=-1)),
        ]),
        "Decision Tree": Pipeline([
            ("prep", preprocessor),
            ("clf",  DecisionTreeClassifier(max_depth=10, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("clf",  RandomForestClassifier(n_estimators=100, max_depth=15,
                                            random_state=42, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("prep", preprocessor),
            ("clf",  xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                eval_metric="logloss", random_state=42,
                n_jobs=-1, verbosity=0,
            )),
        ]),
        "LightGBM": Pipeline([
            ("prep", preprocessor),
            ("clf",  lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                random_state=42, n_jobs=-1, verbose=-1,
            )),
        ]),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HYPERPARAMETER SEARCH SPACES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PARAM_GRIDS = {
    "LightGBM": {
        "clf__n_estimators":      [100, 200, 300],
        "clf__learning_rate":     [0.05, 0.10, 0.15],
        "clf__max_depth":         [4, 6, 8],
        "clf__num_leaves":        [31, 63, 127],
        "clf__min_child_samples": [20, 50],
        "clf__subsample":         [0.8, 1.0],
        "clf__colsample_bytree":  [0.8, 1.0],
    },
    "XGBoost": {
        "clf__n_estimators":      [100, 200, 300],
        "clf__learning_rate":     [0.05, 0.10, 0.15],
        "clf__max_depth":         [4, 6, 8],
        "clf__subsample":         [0.8, 1.0],
        "clf__colsample_bytree":  [0.8, 1.0],
    },
    "Random Forest": {
        "clf__n_estimators":      [100, 200, 300],
        "clf__max_depth":         [None, 15, 25],
        "clf__min_samples_split": [2, 5],
        "clf__max_features":      ["sqrt", "log2"],
    },
}


def tune_model(pipeline, param_grid: dict,
               X_train, y_train, name: str, n_iter: int = 15):
    print(f"\n  [tune] RandomizedSearchCV on {name} (n_iter={n_iter}) ...")
    search = RandomizedSearchCV(
        pipeline, param_grid,
        n_iter=n_iter, cv=CV, scoring="roc_auc",
        n_jobs=-1, random_state=42, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  [tune] Best CV ROC-AUC : {search.best_score_:.4f}")
    print(f"  [tune] Best params     : {search.best_params_}")
    return search.best_estimator_, float(search.best_score_)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_training():
    # 1. Load & prepare
    print("\n[train] Loading and preparing data ...")
    raw  = load_raw()
    df   = engineer_features(clean(raw))
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    # 2. Cross-validation pass
    print("\n[train] -- 5-Fold Cross-Validation (ROC-AUC) --")
    models    = get_base_models(preprocessor)
    cv_scores = {}
    for name, pipe in models.items():
        cv_scores[name] = cv_score(pipe, X_train, y_train, name)

    best_cv_name = max(cv_scores, key=cv_scores.get)
    print(f"\n  Best baseline CV model: {best_cv_name} "
          f"({cv_scores[best_cv_name]:.4f})")

    # 3. Fit all models on full training set
    print("\n[train] -- Training all models --")
    all_metrics = []
    fitted_models = {}
    for name, pipe in models.items():
        print(f"  Fitting {name} ...", end=" ", flush=True)
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe
        m = evaluate(pipe, X_test, y_test, name)
        m["cv_roc_auc"] = cv_scores[name]
        all_metrics.append(m)

    # 4. Hyperparameter tuning
    print("\n[train] -- Hyperparameter Tuning (RandomizedSearchCV) --")
    tune_candidates = ["LightGBM", "XGBoost", "Random Forest"]
    tuned_models = {}
    for name in tune_candidates:
        if name in PARAM_GRIDS:
            fresh_pipe       = get_base_models(preprocessor)[name]
            tuned, tuned_cv  = tune_model(
                fresh_pipe, PARAM_GRIDS[name],
                X_train, y_train, name, n_iter=15,
            )
            tuned.fit(X_train, y_train)
            tuned_models[name] = (tuned, tuned_cv)

    best_tuned_name           = max(tuned_models, key=lambda k: tuned_models[k][1])
    best_tuned_model, best_cv = tuned_models[best_tuned_name]
    print(f"\n  Best tuned model: {best_tuned_name} (CV AUC={best_cv:.4f})")

    m_tuned = evaluate(best_tuned_model, X_test, y_test,
                       f"{best_tuned_name} (Tuned)")
    m_tuned["cv_roc_auc"] = best_cv
    all_metrics.append(m_tuned)

    # 5. Save artifacts
    results_df = pd.DataFrame(all_metrics)
    joblib.dump(best_tuned_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(feature_names,    MODEL_DIR / "feature_names.pkl")
    results_df.to_csv(MODEL_DIR / "model_results.csv", index=False)

    # Save test set for evaluation module
    test_df = X_test.copy()
    test_df["is_canceled"] = y_test.values
    test_df.to_csv(MODEL_DIR / "test_data.csv", index=False)

    print(f"\n[train] Best model saved  -> {MODEL_DIR / 'best_model.pkl'}")
    print(f"[train] Results saved     -> {MODEL_DIR / 'model_results.csv'}")

    return best_tuned_model, results_df


if __name__ == "__main__":
    best, results = run_training()
    print("\n\n[train] -- Final Leaderboard --")
    print(
        results[["model","accuracy","precision","recall","f1","roc_auc","cv_roc_auc"]]
        .sort_values("roc_auc", ascending=False)
        .to_string(index=False)
    )
