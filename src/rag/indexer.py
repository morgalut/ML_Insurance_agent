# src/rag/indexer.py
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def load_corpus(path):
    """Load all markdown docs from folder."""
    files = sorted(glob.glob(os.path.join(path, "*.md")))
    texts = [open(f, encoding="utf-8").read() for f in files]
    return files, texts

def build_tfidf_index(texts, max_features=3000):
    """Fit a simple TF-IDF index."""
    vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=max_features,
        stop_words=None, norm="l2"
    )
    X = vec.fit_transform(texts)
    return vec, X

def retrieve_top_docs(vec, X, query, top_k=3):
    """Retrieve top-k documents by cosine similarity."""
    q = vec.transform([query])
    sims = (q @ X.T).toarray().ravel()
    idx = np.argsort(-sims)[:top_k]
    return idx, sims[idx]
