# src/rag/generate.py
import json
import os
import joblib
import numpy as np
from .indexer import load_corpus, build_tfidf_index, retrieve_top_docs

OUT_RAG = "out/rag"
os.makedirs(OUT_RAG, exist_ok=True)

def render_plan_steps(query, doc_ids, docs, label_prefix="Doc"):
    """Construct a 3-step plan referencing [Doc#]."""
    steps = []
    for i, doc_idx in enumerate(doc_ids[:3], start=1):
        snippet = docs[doc_idx].split("\n")[0][:160]
        steps.append(f"{i}. {snippet.strip()} [{label_prefix}#{doc_idx+1}]")
    return steps



# Lapse Prevention Plans (uses model predictions)

def generate_lapse_prevention_plans(model):
    """Generate 3-step lapse prevention strategies for 3 customers (high/med/low)."""
    from scipy.sparse import issparse

    # Load test data
    X_test, y_test = joblib.load("out/processed/test.joblib")

    # Predict lapse probabilities
    preds = model.predict_proba(X_test)[:, 1] if issparse(X_test) else model.predict_proba(X_test)[:, 1]

    # Select 3 representative customers
    idx_sorted = np.argsort(preds)
    low_idx, med_idx, high_idx = idx_sorted[0], idx_sorted[len(preds)//2], idx_sorted[-1]
    sample_idxs = [low_idx, med_idx, high_idx]

    # Load RAG corpus
    lapse_path = "data/rag/lapse"
    files, docs = load_corpus(lapse_path)
    if not docs:
        raise RuntimeError(f"No documents found in {lapse_path}. Please add markdown files.")
    vec, X = build_tfidf_index(docs)

    results = []
    for i, idx in enumerate(sample_idxs):
        p = float(preds[idx])
        prompt = (
            f"Customer predicted lapse probability = {p:.2f}. "
            "Generate a concise 3-step retention plan based on retrieved strategies."
        )
        doc_ids, _ = retrieve_top_docs(vec, X, prompt, top_k=3)
        steps = render_plan_steps(prompt, doc_ids, docs)
        results.append({
            "case_id": i + 1,
            "probability": p,
            "summary": f"Predicted lapse risk = {p:.2f}. Recommended 3-step plan below.",  # probability in text
            "retrieved_docs": [files[d] for d in doc_ids],
            "plan_steps": steps,
            "citations": [f"[Doc#{d+1}]" for d in doc_ids],
        })

    out_path = os.path.join(OUT_RAG, "lapse_plans.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f" Lapse prevention plans written to {out_path}")
    return results



# Lead Conversion Plans 

def generate_lead_conversion_plans():
    """Generate 3-step personalized conversion plans for synthetic leads."""
    profiles = [
        "Young digital prospect, Region C, price-sensitive smoker curious about health coverage.",
        "Mid-career parent, Region B, through agent channel, concerned about premiums.",
        "Retiree in Region D evaluating annuity options for dependents.",
    ]

    lead_path = "data/rag/lead"
    files, docs = load_corpus(lead_path)
    if not docs:
        raise RuntimeError(f"No documents found in {lead_path}. Please add markdown files.")
    vec, X = build_tfidf_index(docs)

    results = []
    for i, profile in enumerate(profiles, 1):
        prompt = (
            f"Lead profile: {profile}\n"
            "Generate a concise 3-step conversion plan citing relevant strategies."
        )
        doc_ids, _ = retrieve_top_docs(vec, X, prompt, top_k=3)
        steps = render_plan_steps(prompt, doc_ids, docs)
        results.append({
            "lead_id": i,
            "profile": profile,
            "summary": f"Generated 3-step lead conversion plan for profile #{i}.",
            "retrieved_docs": [files[d] for d in doc_ids],
            "plan_steps": steps,
            "citations": [f"[Doc#{d+1}]" for d in doc_ids],
        })

    out_path = os.path.join(OUT_RAG, "lead_plans.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f" Lead conversion plans written to {out_path}")
    return results
