"""
Reproducible training pipeline for the Digital Finance Access Predictor (v2).

What changed from the v1 thesis notebook, and why:
  1. Codes follow the Global Findex 2021 codebook:
     urbanicity 1 = Rural, 2 = Urban; educ 1 = primary or less,
     2 = secondary, 3 = tertiary or more; female 1 = Female, 2 = Male.
  2. `wgt` (a survey design weight) is no longer used as a feature.
  3. Country-level regulation variables are dropped as features. With three
     countries they take only three values, so they duplicate the country
     indicator and cannot measure the effect of regulation.
  4. The router is trained on out-of-fold predictions, not on the base
     models' predictions for their own training rows. A classic
     mixture of experts (each person goes to their own country's expert)
     is added, so the MoE idea gets a fair test.
  5. Models are ranked by ROC-AUC. Accuracy is reported next to the
     majority-class baseline, because about 76% of respondents have an account.
  6. Each country expert is compared with the other models on the SAME
     country's test rows.
  7. Results are also given as 5-fold cross-validated mean and spread.

Usage:
    python train.py --data data/final_combined_data.csv
Outputs go to models/ (artefacts used by app.py) and results/ (tables, charts).
"""

import argparse
import json
import os
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:  # only for testing the pipeline logic without XGBoost
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_XGB = False

SEED = 42
TARGET = "dig_account"
COUNTRIES = ["KEN", "TZA", "UGA"]
BASE_MODELS = ["Pooled", "Harmonized"] + [f"Expert_{c}" for c in COUNTRIES]
INDIVIDUAL_FEATURES = ["female", "age", "educ", "inc_q", "urbanicity", "internet_use"]
COUNTRY_FEATURES = [f"country_{c}" for c in COUNTRIES]
FEATURES = INDIVIDUAL_FEATURES + COUNTRY_FEATURES
GENERAL_MODELS = ["Pooled", "Harmonized", "MoE_CountryGate", "MoE_Router", "LogReg_Baseline"]


# ---------------------------------------------------------------- data
def load_data(path):
    df = pd.read_csv(path)
    df["country_code"] = df["country_code"].astype(str).str.strip().str.upper()
    before = len(df)
    df = df.dropna(subset=INDIVIDUAL_FEATURES + [TARGET]).reset_index(drop=True)
    print(f"Rows: {before} -> {len(df)} after dropping rows with missing values")

    checks = {
        "female": {1, 2}, "educ": {1, 2, 3}, "urbanicity": {1, 2},
        "inc_q": {1, 2, 3, 4, 5}, "internet_use": {0, 1}, TARGET: {0, 1},
    }
    for col, allowed in checks.items():
        found = set(df[col].astype(int).unique())
        if not found <= allowed:
            raise ValueError(f"{col}: unexpected codes {sorted(found - allowed)}")
        df[col] = df[col].astype(int)
    return df


def build_features(df):
    X = df[INDIVIDUAL_FEATURES].copy()
    for c in COUNTRIES:
        X[f"country_{c}"] = (df["country_code"] == c).astype(int)
    return X[FEATURES], df[TARGET].astype(int), df["country_code"]


# -------------------------------------------------------------- models
def make_gbm():
    if HAVE_XGB:
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
            eval_metric="logloss", random_state=SEED, n_jobs=2,
        )
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                          max_iter=300, random_state=SEED)


def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def harmonize(X, y, country):
    """Equal number of rows per country (down-sample to the smallest)."""
    n = country.value_counts().min()
    idx = (pd.Series(country.values, index=X.index)
           .groupby(country.values, group_keys=False)
           .apply(lambda s: s.sample(n, random_state=SEED)).index)
    return X.loc[idx], y.loc[idx]


def fit_base(X, y, country):
    models = {"Pooled": make_gbm().fit(X, y)}
    Xh, yh = harmonize(X, y, country)
    models["Harmonized"] = make_gbm().fit(Xh, yh)
    for c in COUNTRIES:
        m = (country == c).values
        models[f"Expert_{c}"] = make_gbm().fit(X[m], y[m])
    return models


def base_probs(models, X):
    return pd.DataFrame({k: models[k].predict_proba(X)[:, 1] for k in BASE_MODELS},
                        index=X.index)


def routing_labels(probs, y):
    """For each row, the base model whose probability is closest to the truth."""
    err = probs.sub(y, axis=0).abs()
    return err.idxmin(axis=1)


def fit_all(X, y, country):
    """Fit base models, the out-of-fold router and a logistic baseline."""
    # Out-of-fold base-model probabilities for the router's training data
    oof = pd.DataFrame(index=X.index, columns=BASE_MODELS, dtype=float)
    strat = country.astype(str) + "_" + y.astype(str)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, strat):
        fold_models = fit_base(X.iloc[tr], y.iloc[tr], country.iloc[tr])
        oof.iloc[va] = base_probs(fold_models, X.iloc[va]).values

    router_X = pd.concat([X, oof], axis=1)
    router = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                    random_state=SEED, n_jobs=2)
    router.fit(router_X, routing_labels(oof, y))

    return {
        "base": fit_base(X, y, country),
        "router": router,
        "router_columns": router_X.columns.tolist(),
        "logreg": make_logreg().fit(X, y),
    }


def predict_all(bundle, X):
    probs = base_probs(bundle["base"], X)
    routed = bundle["router"].predict(pd.concat([X, probs], axis=1)[bundle["router_columns"]])
    probs["MoE_Router"] = [probs.loc[i, m] for i, m in zip(X.index, routed)]
    # Classic mixture of experts: each person goes to their own country's expert
    gate = X[COUNTRY_FEATURES].idxmax(axis=1).str.replace("country_", "Expert_")
    probs["MoE_CountryGate"] = [probs.loc[i, m] for i, m in zip(X.index, gate)]
    probs["LogReg_Baseline"] = bundle["logreg"].predict_proba(X)[:, 1]
    return probs, pd.Series(routed, index=X.index)


# ------------------------------------------------------------- metrics
def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "N": len(y),
        "ROC_AUC": roc_auc_score(y, p) if y.nunique() == 2 else np.nan,
        "Brier": brier_score_loss(y, p),
        "Balanced_Accuracy": balanced_accuracy_score(y, pred),
        "Accuracy": accuracy_score(y, pred),
        "Majority_Baseline_Accuracy": max(y.mean(), 1 - y.mean()),
        "F1": f1_score(y, pred, zero_division=0),
    }


def evaluate(probs, y, country, female, urbanicity):
    overall, by_country, fairness = [], [], []
    general = GENERAL_MODELS
    for m in general:
        overall.append({"Model": m, **metrics(y, probs[m])})
    for c in COUNTRIES:
        mask = (country == c).values
        for m in general + [f"Expert_{c}"]:
            by_country.append({"Country": c, "Model": m,
                               **metrics(y[mask], probs.loc[mask, m])})
    for name, series, labels in [("Gender", female, {1: "Female", 2: "Male"}),
                                 ("Residence", urbanicity, {1: "Rural", 2: "Urban"})]:
        for code, label in labels.items():
            mask = (series == code).values
            fairness.append({"Attribute": name, "Group": label,
                             **metrics(y[mask], probs.loc[mask, "MoE_Router"])})
    return pd.DataFrame(overall), pd.DataFrame(by_country), pd.DataFrame(fairness)


# --------------------------------------------------------------- plots
def save_charts(cv_summary, by_country, fairness, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = cv_summary.sort_values("ROC_AUC_mean")
    ax.barh(s["Model"], s["ROC_AUC_mean"], xerr=s["ROC_AUC_sd"], color="#1E88E5", capsize=4)
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlim(0.4, 1)
    ax.set_xlabel("ROC-AUC (5-fold CV mean ± sd); 0.5 = no skill")
    ax.set_title("Overall model comparison")
    fig.tight_layout(); fig.savefig(os.path.join(out, "model_comparison_auc.png"), dpi=200)
    plt.close(fig)

    piv = by_country.copy()
    piv["Model"] = piv["Model"].where(~piv["Model"].str.startswith("Expert_"), "Country expert")
    piv = piv.pivot(index="Country", columns="Model", values="ROC_AUC")
    order = ["Pooled", "Harmonized", "Country expert", "MoE_Router", "LogReg_Baseline"]
    piv = piv.drop(columns=["MoE_CountryGate"])  # identical to "Country expert" here
    ax = piv[order].plot(kind="bar", figsize=(9, 4.5), rot=0)
    ax.set_ylim(0.4, 1); ax.set_ylabel("ROC-AUC (test set)")
    ax.set_title("Same rows, same country: is the country expert better?")
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "per_country_auc.png"), dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = fairness["Attribute"] + ": " + fairness["Group"]
    ax.bar(labels, fairness["ROC_AUC"], color=["#8E24AA", "#8E24AA", "#43A047", "#43A047"])
    for i, (v, n) in enumerate(zip(fairness["ROC_AUC"], fairness["N"])):
        ax.text(i, v + 0.01, f"{v:.2f}\nN={n}", ha="center", fontsize=8)
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("ROC-AUC (test set)")
    ax.set_title("Subgroup performance of the MoE router")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fairness_auc.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/final_combined_data.csv")
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    os.makedirs(args.models, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)
    print("Engine:", "XGBoost " + xgb.__version__ if HAVE_XGB
          else "sklearn HistGradientBoosting (TEST ONLY - install xgboost)")

    df = load_data(args.data)
    X, y, country = build_features(df)
    strat = country + "_" + y.astype(str)

    # 1) 5-fold cross-validation of the whole procedure (robust comparison)
    cv_rows = []
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, strat)):
        bundle = fit_all(X.iloc[tr], y.iloc[tr], country.iloc[tr])
        probs, _ = predict_all(bundle, X.iloc[te])
        for m in GENERAL_MODELS:
            cv_rows.append({"Fold": k + 1, "Model": m, **metrics(y.iloc[te], probs[m])})
        print(f"CV fold {k + 1}/5 done")
    cv = pd.DataFrame(cv_rows)
    cv_summary = (cv.groupby("Model")[["ROC_AUC", "Brier", "Balanced_Accuracy", "Accuracy"]]
                  .agg(["mean", "std"]))
    cv_summary.columns = [f"{a}_{'sd' if b == 'std' else b}" for a, b in cv_summary.columns]
    cv_summary = cv_summary.reset_index().sort_values("ROC_AUC_mean", ascending=False)

    # 2) Hold-out split for the detailed tables and the deployed models
    tr_idx, te_idx = train_test_split(X.index, test_size=0.2, random_state=SEED, stratify=strat)
    bundle = fit_all(X.loc[tr_idx], y.loc[tr_idx], country.loc[tr_idx])
    probs, routed = predict_all(bundle, X.loc[te_idx])
    overall, by_country, fairness = evaluate(
        probs, y.loc[te_idx], country.loc[te_idx],
        df.loc[te_idx, "female"], df.loc[te_idx, "urbanicity"])
    routing_share = routed.value_counts().rename_axis("Selected_Model").reset_index(name="Count")

    cv_summary.to_csv(os.path.join(args.results, "cv_model_comparison.csv"), index=False)
    overall.to_csv(os.path.join(args.results, "test_model_comparison.csv"), index=False)
    by_country.to_csv(os.path.join(args.results, "per_country_comparison.csv"), index=False)
    fairness.to_csv(os.path.join(args.results, "fairness_results.csv"), index=False)
    routing_share.to_csv(os.path.join(args.results, "router_selection.csv"), index=False)
    save_charts(cv_summary, by_country, fairness, args.results)

    # 3) Save artefacts for the app
    for name, model in bundle["base"].items():
        if HAVE_XGB:
            model.save_model(os.path.join(args.models, f"{name}.json"))
        else:
            joblib.dump(model, os.path.join(args.models, f"{name}.joblib"))
    joblib.dump(bundle["router"], os.path.join(args.models, "router.joblib"), compress=3)
    info = {
        "features": FEATURES,
        "router_columns": bundle["router_columns"],
        "base_models": BASE_MODELS,
        "country_base_rate": df.groupby("country_code")[TARGET].mean().round(4).to_dict(),
        "codes": {"female": {"1": "Female", "2": "Male"},
                  "urbanicity": {"1": "Rural", "2": "Urban"},
                  "educ": {"1": "Primary or less", "2": "Secondary", "3": "Tertiary or more"},
                  "inc_q": "1 = poorest 20% ... 5 = richest 20% (within country)"},
        "engine": "xgboost" if HAVE_XGB else "sklearn-hgb-test",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "xgboost": xgb.__version__ if HAVE_XGB else None,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    with open(os.path.join(args.models, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    pd.set_option("display.width", 160)
    print("\n5-fold CV (mean ± sd):\n", cv_summary.round(3).to_string(index=False))
    print("\nPer-country (test set):\n", by_country[["Country", "Model", "N", "ROC_AUC",
          "Accuracy", "Majority_Baseline_Accuracy"]].round(3).to_string(index=False))
    print("\nFairness (MoE, test set):\n", fairness[["Attribute", "Group", "N", "ROC_AUC",
          "Accuracy"]].round(3).to_string(index=False))
    print("\nRouter selection:\n", routing_share.to_string(index=False))
    print("\nVersions:", info["versions"])


if __name__ == "__main__":
    main()
