## 📁 Output after running

After you run:

```bash
python src/data.py
```

You’ll have:

```
out/processed/
├── train.joblib          # (X_train_sparse, y_train)
├── val.joblib            # (X_val_sparse, y_val)
├── test.joblib           # (X_test_sparse, y_test)
├── onehot_encoder.joblib
├── scaler.joblib
└── feature_names.joblib
```
