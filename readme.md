

# Deliverable layout (GitHub repo)

### How run 
```sh
pip install -r requirements.txt
```
```sh
python run.py
```

----
```
atidot-assignment/
├─ run.py
├─ requirements.txt            # xgboost, scikit-learn, pandas, numpy, shap, matplotlib, scipy
├─ src/
│  ├─ data.py                  # synthetic generator (+ drift + leakage trap)
│  ├─ split.py                 # strict temporal split + validators
│  ├─ model.py                 # XGBoost train/tune (≤30 trials) + early stopping
│  ├─ metrics.py               # AUC-PR, precision@k, calibration helpers
│  ├─ explain.py               # SHAP global bar plot
│  ├─ rag/
│  │  ├─ indexer.py            # TF-IDF per corpus, deterministic top-k retrieval
│  │  └─ generate.py           # templating + citation insertion, probability-in-prompt
│  └─ utils.py                 # seeds, timers, I/O, determinism checks
├─ data/
│  └─ rag/
│     ├─ lapse/                # 5–6 tiny markdown docs you write with ChatGPT text
│     │  ├─ 01_grace_period.md
│     │  ├─ 02_agent_outreach.md
│     │  ├─ 03_payment_plans.md
│     │  ├─ 04_loyalty_discounts.md
│     │  ├─ 05_seasonality.md
│     │  └─ 06_smoker_coaching.md
│     └─ lead/                 # 5–6 tiny markdown docs for lead conversion
│        ├─ 01_segment_msg.md
│        ├─ 02_cadence.md
│        ├─ 03_objections.md
│        ├─ 04_value_props.md
│        ├─ 05_trial_discount.md
│        └─ 06_channel_guidelines.md
├─ out/                        # created by run.py
│  ├─ dataset.csv
│  ├─ split_manifest.json
│  ├─ model.xgb
│  ├─ metrics.json
│  ├─ shap_global_bar.png
│  ├─ rag/
│  │  ├─ lapse_plans.jsonl     # 3 customers (high/median/low) with prob + citations
│  │  └─ lead_plans.jsonl      # 3 synthetic leads with citations
│  └─ logs.txt
└─ DISCUSSION.md               # brief notes: leakage, drift, split, metrics, SHAP, limits
```

---

# Step-by-step implementation

## 0) Determinism & runtime guardrails

* Set **all seeds** (`numpy`, `random`, `scikit-learn`, `xgboost`) to a fixed `SEED = 42`.
* Disable multithread explosion: set `n_jobs=4` (or fewer) for portability.
* Put timers around each stage; if total time risks >5 minutes, fall back to fewer trials or smaller TF-IDF vocab.

## 1) Generate synthetic temporal data (+drift +leakage trap)

**Specs**

* ~2,000 policies × 12 months ≈ 24,000 rows.
* Columns: `policy_id, month, age, tenure_m, premium, coverage, region, has_agent, is_smoker, dependents`.
* **Target:** `lapse_next_3m` (if lapse occurs within next 3 months).
* **Drift:** After `month >= '2023-07'`, nudge distributions (e.g., premium inflation, changed churn base rate).
* **Leakage trap:** Include something like `post_event_payment_holiday` (only non-null after a delinquency event that *depends on future information*). **Explicitly drop it** from the model features.

**How**

* Create a monthly panel from a start date (e.g., `2022-08` to `2023-07` inclusive, 12 months).
* Simulate base *propensity to lapse* from interpretable drivers:

  * Higher lapse risk: **high premium / low coverage (bad price-value)**, **no agent**, **smoker**, **young tenure**, certain **regions**.
  * Lower lapse risk: **has_agent**, **long tenure**, **discounts/loyalty** (can be proxied via feature transformations).
* Compute `will_lapse_in_next_3m` per policy-month using a logistic transform, then roll it forward 3 months to define `lapse_next_3m`.

**Runtime tip:** vectorize with NumPy/Pandas (no loops). Save to `out/dataset.csv`.

## 2) Strict temporal split (+validation)

**Split**

* Choose train: first ~70% months, val: next ~15%, test: last ~15%.

  * Example with 12 months: **train** months 1–8, **val** 9–10, **test** 11–12.
* **Programmatic validators:**

  * Assert `max(train.month) < min(val.month) < min(test.month)`.
  * Assert no `policy_id,month` overlaps.
  * Optionally assert distribution shift exists (drift check) by a simple statistic (e.g., premium mean shift after 2023-07).

**Leakage guard**

* Maintain `ALL_FEATURES` and `LEAK_TRAPS = ['post_event_payment_holiday']`.
* Derive `X = df[ALL_FEATURES - LEAK_TRAPS]`.
* Assert traps absent in the final training matrix.

**Write** a small `split_manifest.json` into `out/` with month ranges and SHA-1 hashes of index lists for reproducibility.

## 3) Model: XGBoost classifier with early stopping + light tuning (≤30 trials)

**Why XGBoost?** Fast, good on tabular, supports early stopping and handles class imbalance.

**Setup**

* Use `xgboost.XGBClassifier(tree_method='hist', eval_metric='aucpr', random_state=SEED, n_estimators=300, max_depth∈[3..7], learning_rate∈[0.03..0.2], subsample∈[0.6..1.0], colsample_bytree∈[0.6..1.0], min_child_weight∈[1..8])`.
* **Tuning:** RandomizedSearch (sklearn) or Optuna with ≤30 trials. Stop each trial early with `early_stopping_rounds=30` on the **validation** set.
* **Class imbalance:** use `scale_pos_weight` ≈ `neg/pos` computed on the train set.

**Artifacts**

* Save best model to `out/model.xgb`.
* Save `metrics.json` with `auc_pr_train/val/test` and `precision_at_k` for k=1% and 5% on **test**.

## 4) Metrics: AUC-PR & precision@k (k=1%, 5%)

**AUC-PR**

* Use `sklearn.metrics.average_precision_score(y_true, y_scores)` on splits.

**precision@k**

* For each `k`:

  * `n = ceil(k% * len(test))`
  * Select top-n by predicted probability.
  * Compute precision = positives_in_top_n / n.
* Log counts for transparency (how many flagged, how many truly lapsed).

## 5) Explainability: Global SHAP bar plot

* Use `shap.TreeExplainer(model).shap_values(X_sample)` where `X_sample` is a **small** stratified subsample (e.g., 2k rows) to keep runtime down.
* Aggregate mean |SHAP| per feature and plot a horizontal bar chart to `out/shap_global_bar.png`.
* Briefly discuss in `DISCUSSION.md` which signals drive risk (e.g., high premium / low coverage ratio, tenure, agent).

> Note: SHAP can be the slowest part; keep `X_sample` small and skip beeswarm. One static bar plot is enough.

## 6) RAG: Two tiny TF-IDF corpora + grounded generation

**Corpora**

* You write 5–6 tiny markdown docs per corpus under `data/rag/…` (you can draft them with ChatGPT then paste).

  * *Lapse corpus* topics they suggested: **grace period**, **agent outreach**, **payment plans**, **loyalty discounts**, **seasonality/smoker coaching**.
  * *Lead corpus*: **segment messaging**, **cadence**, **objection handling**, **value props**, **trial/discount guidelines**, **channel do’s/don’ts**.

**Indexer (deterministic)**

* Build TF-IDF with `TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=3000, norm='l2')`.
* Represent each doc as a row; cosine similarity via dot product on L2-normed vectors.
* Keep a fixed doc order; assign `[Doc#]` by filename order so citations are stable.

**Generation (no external LLM keys)**

* **No-API default**: a tiny rule-based + template generator that:

  * Retrieves top-k docs (e.g., k=3).
  * Extracts the 1–2 most relevant sentences per doc.
  * Renders a *3-step plan* using a prompt template with slot-fill + paraphrase heuristics (simple synonyms, lists).
* (Optional) If you want, allow an **API-based path** (OpenAI, etc.) but keep it **off by default**, and gate it behind `--use_api false`.

**Lapse prevention branch (probability-in-prompt)**

* Pick **3 test customers**: highest risk, median risk near threshold, and low risk.

* For each, compute `p = model.predict_proba(X_test)`.

* Inject `p` explicitly into the prompt, e.g.:

  > Customer **{policy_id}** predicted **lapse risk = 0.41** (41%). Create a concise 3-step plan to prevent lapse. Cite retrieved docs as `[Doc#]`.

* Ensure the final answer includes `[Doc#]` citations from the retrieved sources.

**Lead conversion branch**

* Define 3 synthetic leads (e.g., *Young digital prospect in Region C, price-sensitive and curious about smoker surcharge*; *Mid-career parent via agent channel with coverage gap objection*; *Retiree comparing annuity options*).
* Generate 3-step conversion plans grounded with `[Doc#]` citations.

**Outputs**

* Write JSONL records to `out/rag/lapse_plans.jsonl` and `out/rag/lead_plans.jsonl` with fields:

  * `case_id`, `retrieved_docs: [doc_ids]`, `probability` (for lapse), `plan_steps: [..]`, `citations: ["[Doc#1]", "[Doc#2]"]`.

## 7) Anti-Triviality verifications (must-pass checks)

In `run.py`, after each stage, run explicit assertions and write proof into logs:

1. **Data leakage safeguard**

   * Assert no `LEAK_TRAPS` are present in training columns.
   * (Optional) Train a “cheat model” with the trap included on a small subset; if AUC-PR skyrockets unrealistically → log a warning to show you *detected* it and *excluded* it.

2. **Temporal integrity**

   * Assertions on month ordering per split and non-overlap.
   * Persist a `split_manifest.json` with month windows and sample SHA-1 of indices.

3. **RAG faithfulness**

   * For every generated answer, assert `len(citations) >= 1` and each `[Doc#]` exists in retrieved set.
   * Optionally verify overlap of TF-IDF top docs and citations ≥1.

4. **Probability-in-prompt (lapse branch)**

   * Assert that each lapse plan JSON includes a numeric `probability` and that the rendered text contains that percentage.

5. **Determinism & reproducibility**

   * Re-index corpora in a fixed order.
   * Seed TF-IDF’s tokenization (deterministic by default if input order is fixed).
   * Save `requirements.txt` with pinned or compatible versions; set seeds.
   * Optional: re-run a tiny smoke re-execution of predictions on the same mini-batch and assert identical outputs.

6. **Runtime discipline**

**`run.py`** orchestrates:

1. Seed + timers.
2. Generate dataset (+drift +leak trap) → `out/dataset.csv`.
3. Split (validate) → manifest JSON.
4. Train/tune XGB (≤30 trials) with early stopping.
5. Evaluate AUC-PR + precision@1%/5% on test → `metrics.json`.
6. SHAP global bar → `shap_global_bar.png`.
7. RAG:

   * Build two TF-IDF indices (lapse, lead).
   * **Lapse:** pick 3 test customers (high/median/low risk), include **probability** explicitly in each prompt, retrieve top-k, output **3-step plan with `[Doc#]` citations**.
   * **Lead:** define 3 synthetic leads, retrieve, output plans with citations.
8. Write everything under `out/`, print a short summary, and exit.

---

# What to say in `DISCUSSION.md` (brief but targeted)

* **Data generation:** how features influence lapse risk; how drift was injected (e.g., premium↑ and base propensity↑ after 2023-07).
* **Leakage trap:** what it is (`post_event_payment_holiday`) and why excluding it matters; (optional) a quick sanity showing a “cheat model” performs unrealistically if included.
* **Temporal integrity:** exact month ranges for train/val/test and programmatic checks.
* **Model & metrics:** why AUC-PR (class imbalance), report test AUC-PR and precision@k; quick interpretation of **SHAP**.
* **RAG:** how TF-IDF retrieval works, how `[Doc#]` mapping is deterministic, and *probability-in-prompt* placement.
* **Determinism & runtime:** seeds, thread limits, capped trials, doc ordering; measured wall-clock time on your laptop.
* **Limitations:** synthetic nature; TF-IDF simplicity; potential improvements if more time (calibration, cost curves, richer retrieval, guardrails).

---

# Practical tips to hit the 5-minute runtime

* Keep the dataset small but sufficient (~24k rows).
* XGBoost with `tree_method="hist"` and early stopping is fast; cap trials at 20–30.
* SHAP on a small subsample (1–2k rows) with **global mean |SHAP|** only.
* TF-IDF indexing is instant on 10–12 documents.
* Avoid any network calls by default.

---


