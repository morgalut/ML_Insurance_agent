# src/split.py
import pandas as pd, hashlib, json, os

def validate_temporal_split(df, out_path="out/split_validation.json"):
    months = sorted(df["month"].unique())
    n = len(months)
    tr, va, te = months[:int(0.7*n)], months[int(0.7*n):int(0.85*n)], months[int(0.85*n):]
    dtr, dva, dte = df[df["month"].isin(tr)], df[df["month"].isin(va)], df[df["month"].isin(te)]

    # Assertions
    assert dtr["month"].max() < dva["month"].min() < dte["month"].min(), "Temporal order violated!"
    for name, d in [("train", dtr), ("val", dva), ("test", dte)]:
        assert not d.duplicated(subset=["policy_id","month"]).any(), f"Duplicate {name}"

    def sha1_of(df):
        return hashlib.sha1(pd.util.hash_pandas_object(df[["policy_id","month"]], index=False).values).hexdigest()

    manifest = {
        "train_hash": sha1_of(dtr),
        "val_hash": sha1_of(dva),
        "test_hash": sha1_of(dte),
        "train_months": tr,
        "val_months": va,
        "test_months": te,
        "counts": {"train": len(dtr), "val": len(dva), "test": len(dte)}
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f: json.dump(manifest, f, indent=2)
    print(f" Temporal split validated and saved → {out_path}")
