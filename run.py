# run.py
import time
from src.data import preprocess_and_split
from src.model import train_xgb_model
from src.explain import generate_shap_plot
from src.rag.generate import generate_lapse_prevention_plans, generate_lead_conversion_plans
from src.utils import StageTimer

from src.verify import run_all_verifications

if __name__ == "__main__":
    t0 = time.time()

    print("=== STEP 1–2: Data Preprocessing ===")
    with StageTimer("preprocess"):
        preprocess_and_split()

    print("\n=== STEP 3–4: Model Training & Metrics ===")
    with StageTimer("train"):
        model, metrics = train_xgb_model()

    print("\n=== STEP 5: Explainability ===")
    with StageTimer("explain"):
        generate_shap_plot()

    print("\n=== STEP 6: RAG Plan Generation ===")
    with StageTimer("rag"):
        generate_lapse_prevention_plans(model)
        generate_lead_conversion_plans()

    total_sec = time.time() - t0
    print(f"\n=== STEP 7: Verifications (total runtime {total_sec:.2f}s) ===")
    run_all_verifications(total_sec)

    print("\n🎯 All steps complete — outputs saved under /out/")
