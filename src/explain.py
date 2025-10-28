# src/explain.py
import os
import re
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import xgboost as xgb
from scipy.sparse import issparse

OUT_DIR = "out"



# Utility: dense sampling for SHAP performance

def _dense_sample(X, n=2000, seed=42):
    """Convert sparse matrix to dense sample (for SHAP speed)."""
    rng = np.random.default_rng(seed)
    n = min(n, X.shape[0])
    if issparse(X):
        Xd = X[:n].toarray()
    else:
        idx = rng.choice(X.shape[0], size=n, replace=False)
        Xd = X[idx]
    return Xd



# Utility: Deep-clean XGBoost booster JSON

def _fix_booster_json(booster):
    """
    Fixes malformed base_score='[5E-1]' (string form of 0.5) appearing anywhere
    inside the model JSON. Performs recursive replacement before reloading.
    """
    tmp_path = os.path.join(OUT_DIR, "booster_temp.json")
    booster.save_model(tmp_path)

    with open(tmp_path, "r", encoding="utf-8") as f:
        raw = f.read()

    if "[5E-1]" in raw:
        print(" Fixing malformed base_score='[5E-1]' → '0.5' (deep recursive)")
        raw = re.sub(r'"\[5E-1\]"', '"0.5"', raw)
        raw = raw.replace("[5E-1]", "0.5")

        try:
            data = json.loads(raw)

            def fix_values(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, str) and "[5E-1]" in v:
                            obj[k] = "0.5"
                        else:
                            fix_values(v)
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        if isinstance(v, str) and "[5E-1]" in v:
                            obj[i] = "0.5"
                        else:
                            fix_values(v)

            fix_values(data)
            raw = json.dumps(data)
        except Exception as e:
            print(f" JSON reparse failed; using regex-only patch: {e}")

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(raw)

    clean = xgb.Booster()
    clean.load_model(tmp_path)
    return clean



# Utility: Global SHAP bar chart

def _global_bar(feature_names, shap_values, path):
    """Save top 15 global SHAP importances to bar chart."""
    shap_abs = np.abs(shap_values).mean(axis=0)
    pairs = list(zip(feature_names, shap_abs))
    pairs.sort(key=lambda t: t[1], reverse=True)
    top = pairs[:15]

    plt.figure(figsize=(8, 6))
    plt.barh([p[0] for p in top][::-1], [p[1] for p in top][::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Global Feature Importance (Top 15)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



# Main explainability routine

def generate_shap_plot():
    """
    Compute global SHAP importances robustly and save plot to out/shap_global_bar.png.
    Handles malformed model JSON and falls back to KernelExplainer when necessary.
    """
    # ---------------- Load artifacts ----------------
    model_path = os.path.join(OUT_DIR, "model.joblib")
    model = joblib.load(model_path)
    X_train, y_train = joblib.load(os.path.join(OUT_DIR, "processed", "train.joblib"))
    feature_names = joblib.load(os.path.join(OUT_DIR, "processed", "feature_names.joblib"))

    print(" Loaded model and training split.")

    # ---------------- Prepare dense sample ----------------
    X_dense = _dense_sample(X_train, n=2000, seed=42)
    print(f" Computing SHAP values on {X_dense.shape[0]} samples and {X_dense.shape[1]} features...")

    bg_size = min(100, X_dense.shape[0])
    bg_idx = np.random.default_rng(42).choice(X_dense.shape[0], size=bg_size, replace=False)
    background = X_dense[bg_idx]

    # Silence warnings
    warnings.filterwarnings("ignore", message="feature_perturbation='interventional'")
    warnings.filterwarnings("ignore", message="base_score")

    # ---------------- Try TreeExplainer first ----------------
    try:
        booster = model.get_booster()
        booster_clean = _fix_booster_json(booster)
        print(" Booster normalized via in-memory JSON fix.")

        try:
            explainer = shap.TreeExplainer(
                booster_clean,
                data=background,
                feature_perturbation="interventional",
                model_output="probability",
            )
            shap_values = explainer.shap_values(X_dense)
            print(" SHAP TreeExplainer computed with interventional background.")

        except Exception as e_interv:
            print(f" Interventional TreeExplainer failed: {e_interv}")
            print(" Retrying with feature_perturbation='auto' ...")

            try:
                explainer = shap.TreeExplainer(
                    booster_clean,
                    data=background,
                    feature_perturbation="auto",
                    model_output="probability",
                )
                shap_values = explainer.shap_values(X_dense)
                print(" SHAP TreeExplainer computed with auto perturbation.")
            except Exception as e_auto:
                raise RuntimeError(f"TreeExplainer (auto) failed: {e_auto}")

    # ---------------- Fallback: KernelExplainer ----------------
    except Exception as e_tree:


        # Reduced background and eval for speed
        bg_kernel = background[:50]
        f = lambda data: model.predict_proba(data)[:, 1]
        explainer = shap.KernelExplainer(f, bg_kernel)

        X_eval = shap.sample(X_dense, 300, random_state=42)
        shap_values = explainer.shap_values(X_eval, nsamples=100)
        X_dense = X_eval
        print(" SHAP KernelExplainer computed on reduced sample (300×features).")

    # ---------------- Plot and save ----------------
    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "shap_global_bar.png")
    _global_bar(feature_names, shap_values, out_png)
    print(f" SHAP plot saved to {out_png}")



# Entry point

if __name__ == "__main__":
    generate_shap_plot()
