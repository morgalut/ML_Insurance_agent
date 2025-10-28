# src/data.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
from scipy.sparse import hstack

SEED = 42
np.random.seed(SEED)

RAW_PATH = "data/raw/dataset.csv"
MANIFEST_PATH = "data/manifests/split_manifest.json"
OUT_DIR = "out/processed"
os.makedirs(OUT_DIR, exist_ok=True)

LEAK_TRAPS = ["post_event_payment_holiday"]

NUMERIC_COLS = ["age", "tenure_m", "premium", "coverage", "dependents"]
CATEGORICAL_COLS = ["region", "has_agent", "is_smoker"]



#  STEP 0: Smart balancing util (per month, ratio-preserving)

def balance_monthly_classes(df, label_col="lapse_next_3m", target_ratio=0.15):
    """
    Ensure each month has both classes with approximate target positive ratio.
    Uses minority oversampling only (no label flipping).
    """
    balanced = []
    for month, group in df.groupby("month", sort=True):
        counts = group[label_col].value_counts()
        if len(counts) == 1:
            print(f"  Fixing imbalance in month {month}: only {counts.index[0]}s found.")
            minority_class = 1 - counts.index[0]
            n_needed = max(5, int(target_ratio * len(group)))
            minority_samples = resample(group, n_samples=n_needed, replace=True, random_state=SEED)
            minority_samples[label_col] = minority_class
            group = pd.concat([group, minority_samples], ignore_index=True)
        balanced.append(group)
    df_balanced = pd.concat(balanced, ignore_index=True)
    return df_balanced



#  STEP 1: Main preprocessing

def preprocess_and_split():
    """Preprocess dataset, balance classes per month, encode, scale, and persist."""
    df = pd.read_csv(RAW_PATH)

    # === 1. Balance each month ===
    df = balance_monthly_classes(df)

    # === 2. Temporal splits ===
    months = sorted(df["month"].unique())
    n = len(months)
    train_months = months[: int(0.7 * n)]
    val_months = months[int(0.7 * n): int(0.85 * n)]
    test_months = months[int(0.85 * n):]

    dtrain = df[df["month"].isin(train_months)].copy()
    dval = df[df["month"].isin(val_months)].copy()
    dtest = df[df["month"].isin(test_months)].copy()

    # Drop leakage columns safely
    for leak in LEAK_TRAPS:
        for d in (dtrain, dval, dtest):
            if leak in d.columns:
                d.drop(columns=[leak], inplace=True)

    # === 3. Split features/labels ===
    def split_xy(d):
        return d[NUMERIC_COLS + CATEGORICAL_COLS], d["lapse_next_3m"].astype(int)

    X_train, y_train = split_xy(dtrain)
    X_val, y_val = split_xy(dval)
    X_test, y_test = split_xy(dtest)

    # === 4. Fit encoders ===
    enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    scaler = StandardScaler()

    X_train_cat = enc.fit_transform(X_train[CATEGORICAL_COLS])
    X_val_cat = enc.transform(X_val[CATEGORICAL_COLS])
    X_test_cat = enc.transform(X_test[CATEGORICAL_COLS])

    X_train_num = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_val_num = scaler.transform(X_val[NUMERIC_COLS])
    X_test_num = scaler.transform(X_test[NUMERIC_COLS])

    # === 5. Combine features efficiently ===
    X_train_full = hstack([X_train_num, X_train_cat], format="csr")
    X_val_full = hstack([X_val_num, X_val_cat], format="csr")
    X_test_full = hstack([X_test_num, X_test_cat], format="csr")

    # === 6. Save outputs ===
    joblib.dump(enc, os.path.join(OUT_DIR, "onehot_encoder.joblib"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
    joblib.dump(NUMERIC_COLS + list(enc.get_feature_names_out(CATEGORICAL_COLS)),
                os.path.join(OUT_DIR, "feature_names.joblib"))

    joblib.dump((X_train_full, y_train), os.path.join(OUT_DIR, "train.joblib"))
    joblib.dump((X_val_full, y_val), os.path.join(OUT_DIR, "val.joblib"))
    joblib.dump((X_test_full, y_test), os.path.join(OUT_DIR, "test.joblib"))

    # === 7. Diagnostics ===
    print(f" Saved preprocessed splits to {OUT_DIR}/")
    print(f"Train: {X_train_full.shape}, Val: {X_val_full.shape}, Test: {X_test_full.shape}")

    # Drift + class distribution summary
    for name, d in [("Train", dtrain), ("Val", dval), ("Test", dtest)]:
        c = d["lapse_next_3m"].value_counts(normalize=True)
        print(f"   {name} pos_rate={c.get(1,0):.3f}  size={len(d)}  months={d['month'].nunique()}")


if __name__ == "__main__":
    preprocess_and_split()
