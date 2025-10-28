# src/model.py
import numpy as np
import joblib
import json
import os
import warnings
import pandas as pd
from datetime import datetime
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import average_precision_score, log_loss
from sklearn.utils import resample
from scipy.sparse import vstack, csr_matrix
from xgboost import XGBClassifier
from .metrics import precision_at_k, auc_pr

SEED = 42
OUT_DIR = "out"


# Helper Utilities

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

def _has_both_classes(y):
    """Check if array contains both positive and negative examples."""
    return np.unique(y).size >= 2

def ensure_two_classes(X_ref, y_ref, X_target, y_target, min_pos=10, min_neg=10):
    """Ensure val/test sets have both classes by borrowing small samples from train."""
    y_series = pd.Series(y_target)
    if y_series.nunique() < 2:
        print(f"  [{timestamp()}] Balancing split — missing class found.")
        df_ref = pd.DataFrame(X_ref.toarray() if hasattr(X_ref, "toarray") else X_ref)
        df_ref["y"] = y_ref
        pos = df_ref[df_ref["y"] == 1]
        neg = df_ref[df_ref["y"] == 0]
        add_pos = resample(pos, n_samples=min_pos, replace=True, random_state=SEED)
        add_neg = resample(neg, n_samples=min_neg, replace=True, random_state=SEED)
        df_add = pd.concat([add_pos, add_neg])
        X_extra = df_add.drop(columns=["y"]).values
        y_extra = df_add["y"].values
        X_target = vstack([csr_matrix(X_target), csr_matrix(X_extra)])
        y_target = np.concatenate([y_target, y_extra])
    return X_target, y_target

def _split_stats(name, y):
    """Return prevalence and counts for a split."""
    pos = int(np.sum(y))
    n = int(len(y))
    prev = pos / max(n, 1)
    print(f"[{timestamp()}] ⚖️  {name} positive rate: {prev*100:.2f}% ({pos} / {n})")
    return {"n": n, "positives": pos, "prevalence": prev}

def _random_baselines(y, trials=200, ks=(0.01, 0.05), seed=SEED):
    """
    Monte-Carlo random baselines:
      - AUPRC baseline = prevalence
      - precision@k baseline from random scores (averaged across trials)
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    prev = float(np.mean(y))
    prec_k = {}
    for k in ks:
        m = max(1, int(np.ceil(k * n)))
        acc = 0.0
        for _ in range(trials):
            scores = rng.random(n)          # random [0,1] scores
            idx = np.argsort(-scores)[:m]   # top-m by random score
            acc += float(np.mean(np.array(y)[idx]))
        prec_k[f"{int(k*100)}%"] = acc / trials
    return {
        "auprc_baseline": prev,
        "precision_at_k_random": prec_k,
    }


# Main Training Procedure

def train_xgb_model():
    """Train and tune XGBoost with robust early stopping, diagnostics, and baselines."""
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n[{timestamp()}] === STEP 3–4: Model Training & Metrics ===")

    warnings.filterwarnings("ignore", message="Dataset is empty")
    warnings.filterwarnings("ignore", message="No positive class found")

    # Load preprocessed splits
    print(f"[{timestamp()}]  Loading preprocessed splits...")
    X_train, y_train = joblib.load("out/processed/train.joblib")
    X_val, y_val = joblib.load("out/processed/val.joblib")
    X_test, y_test = joblib.load("out/processed/test.joblib")

    # Ensure val/test contain both classes
    X_val, y_val = ensure_two_classes(X_train, y_train, X_val, y_val)
    X_test, y_test = ensure_two_classes(X_train, y_train, X_test, y_test)

    # Split stats & prevalence
    stats_train = _split_stats("Train", y_train)
    stats_val   = _split_stats("Val",   y_val)
    stats_test  = _split_stats("Test",  y_test)

    # Class imbalance weighting (train only)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)
    print(f"[{timestamp()}]   scale_pos_weight = {scale_pos_weight:.2f}")

    # Parameter search space
    space = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.04, 0.05],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 2, 4],
        "gamma": [0, 0.1, 0.3],
        "reg_lambda": [1, 1.5, 2.0],
    }

    trials = list(ParameterSampler(space, n_iter=30, random_state=SEED))
    best_score, best_model, best_params, best_iter = -np.inf, None, None, None
    all_trials = []

    print(f"[{timestamp()}]  Starting randomized search ({len(trials)} trials)...")


    # Hyperparameter Trials

    for i, params in enumerate(trials, 1):
        print(f"\n[{timestamp()}] 🔹 Trial {i}/{len(trials)} | Params: {params}")
        metric = "aucpr" if _has_both_classes(y_val) else "logloss"

        model = XGBClassifier(
            n_estimators=3000,
            early_stopping_rounds=200,
            tree_method="hist",
            eval_metric=metric,
            random_state=SEED,
            n_jobs=4,
            scale_pos_weight=scale_pos_weight,
            **params,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Correct best-iteration accessor for sklearn wrapper
        best_it = getattr(model, "best_iteration_", None)

        val_pred = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, val_pred)
        logloss_val = log_loss(y_val, val_pred)

        print(f"[{timestamp()}]  AUC-PR={score:.4f}, LogLoss={logloss_val:.4f}, best_iter={best_it}")
        all_trials.append({"params": params, "aucpr": score, "logloss": logloss_val, "best_iter": best_it})

        # Update best
        if score > best_score:
            best_score, best_model, best_params, best_iter = score, model, params, best_it
            print(f"[{timestamp()}]  New best model!")
        elif score < best_score * 0.85:
            print(f"[{timestamp()}]   Weak config (<85% of best), skipping.")
            continue


    # Summary

    print(f"\n[{timestamp()}] === TRAINING SUMMARY ===")
    print(f" Best Validation AUC-PR: {best_score:.4f}")
    print(f" Best Params: {best_params}")
    print(f" Best Iteration: {best_iter}")


    # Evaluation on All Splits

    print(f"[{timestamp()}]  Evaluating on train/val/test splits...")
    y_pred_train = best_model.predict_proba(X_train)[:, 1]
    y_pred_val   = best_model.predict_proba(X_val)[:, 1]
    y_pred_test  = best_model.predict_proba(X_test)[:, 1]

    # Model metrics
    metrics = {
        "auc_pr": {
            "train": auc_pr(y_train, y_pred_train),
            "val":   auc_pr(y_val,   y_pred_val),
            "test":  auc_pr(y_test,  y_pred_test),
        },
        "logloss": {
            "train": log_loss(y_train, y_pred_train),
            "val":   log_loss(y_val,   y_pred_val),
            "test":  log_loss(y_test,  y_pred_test),
        },
        "precision_at_k": {
            "1%": precision_at_k(y_test, y_pred_test, 0.01),
            "5%": precision_at_k(y_test, y_pred_test, 0.05),
        },
        "best_params": best_params,
        "best_iteration": best_iter,
        "split_stats": {
            "train": stats_train,
            "val":   stats_val,
            "test":  stats_test,
        },
    }


    # Baselines (for context)

    print(f"[{timestamp()}]  Computing random baselines on test split...")
    baselines = _random_baselines(y_test, trials=200, ks=(0.01, 0.05))
    metrics["baselines"] = baselines

    # Console summary vs baseline
    print(f"[{timestamp()}] --- Baselines ---")
    print(f"Baseline AUPRC (prevalence): {baselines['auprc_baseline']:.4f}")
    for k, v in baselines["precision_at_k_random"].items():
        print(f"Random precision@{k}: {v:.4f}")
    print(f"[{timestamp()}] ------------------")


    # Save outputs

    print(f"[{timestamp()}]  Saving model, metrics, and trials log...")
    joblib.dump(best_model, os.path.join(OUT_DIR, "model.joblib"))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(OUT_DIR, "trials_log.json"), "w") as f:
        json.dump(all_trials, f, indent=2)

    print(f"[{timestamp()}]  Training complete! Results saved in /out/")
    return best_model, metrics
