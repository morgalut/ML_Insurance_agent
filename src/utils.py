# src/utils.py
import os, json, time, random, hashlib
import numpy as np

SEED = 42

def set_global_seed(seed: int = SEED):
    import numpy as _np
    import random as _random
    try:
        import xgboost as _xgb  # noqa: F401
    except Exception:
        pass
    _np.random.seed(seed)
    _random.seed(seed)

class StageTimer:
    def __init__(self, name: str):
        self.name = name
        self.start_t = None
        self.elapsed = 0.0
    def __enter__(self):
        self.start_t = time.time(); return self
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.time() - self.start_t

def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def sha1_of_pairs(pairs):
    h = hashlib.sha1()
    for a, b in pairs:
        h.update(str(a).encode()); h.update(str(b).encode())
    return h.hexdigest()

def assert_or_raise(condition: bool, msg: str):
    if not condition:
        raise AssertionError(msg)
