"""
hallucination_check.py
PolicyPal AI — NLI-based hallucination & factual-consistency checker.
"""

import argparse, json, re
from pathlib import Path
from typing import Optional
from transformers import pipeline

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

LABEL_MAP = {
    "ENTAILMENT": "entailed",
    "NEUTRAL": "neutral",
    "CONTRADICTION": "hallucinated"
}

Path("results").mkdir(exist_ok=True)


# ================================
# NLI CHECKER
# ================================

class HallucinationChecker:
    def __init__(self, model_name: str = NLI_MODEL):
        print(f"[NLI] Loading {model_name} ...")
        self.nli = pipeline("text-classification", model=model_name, device=-1)

    # 🔥 improved for legal text
    def split_sentences(self, text: str):
        raw = re.split(r"[.!?]+", text.strip())

        clean = []
        for s in raw:
            s = s.strip()

            # skip very short fragments
            if len(s) < 20:
                continue

            # skip legal structural noise
            if any(x in s.lower() for x in [
                "section", "clause", "sub-section",
                "explanation", "act", "amendment"
            ]):
                continue

            clean.append(s)

        return clean

    def check_sentence(self, sentence: str, context: str):
        try:
            # 🔥 shorter context → better signal
            context = context[:300]

            result = self.nli(
                f"Context: {context} Sentence: {sentence}",
                truncation=True,
                max_length=512
            )

            label = result[0]["label"].upper()
            return LABEL_MAP.get(label, "neutral")

        except Exception:
            return "neutral"

    def check_response(self, hypothesis: str, context: str):

        if not hypothesis or len(hypothesis.strip()) < 10:
            return {
                "sentences": [],
                "hallucination_rate": 0.0,
                "verdict": "ok"
            }

        sentences = self.split_sentences(hypothesis)

        if not sentences:
            return {
                "sentences": [],
                "hallucination_rate": 0.0,
                "verdict": "ok"
            }

        verdicts = []

        for sent in sentences:
            label = self.check_sentence(sent, context)
            verdicts.append({
                "sentence": sent,
                "label": label
            })

        n_hallucinated = sum(1 for v in verdicts if v["label"] == "hallucinated")
        rate = round(n_hallucinated / len(verdicts), 4)

        if rate > 0.3:
            overall = "flagged"
        elif rate > 0.1:
            overall = "warning"
        else:
            overall = "ok"

        return {
            "sentences": verdicts,
            "n_total": len(verdicts),
            "n_hallucinated": n_hallucinated,
            "hallucination_rate": rate,
            "verdict": overall,
        }


# ================================
# EVALUATION
# ================================

def run_hallucination_eval(eval_file: str, rag_context_file: Optional[str] = None):

    with open(eval_file, encoding="utf-8") as f:
        eval_data = json.load(f)

    contexts = {}

    if rag_context_file and Path(rag_context_file).exists():
        with open(rag_context_file, encoding="utf-8") as f:
            contexts = json.load(f)

    checker = HallucinationChecker()
    annotated = []
    all_rates = []

    for rec in eval_data["records"]:

        ctx = contexts.get(rec["record_id"], rec["reference"])

        result = checker.check_response(rec["hypothesis"], ctx)

        all_rates.append(result["hallucination_rate"])
        annotated.append({**rec, "hallucination": result})

        verdict_sym = {
            "ok": "✅",
            "warning": "⚠️",
            "flagged": "❌"
        }.get(result["verdict"], "?")

        print(f"{rec['record_id']:<15} rate={result['hallucination_rate']:.2f} {verdict_sym}")

    import statistics

    macro_rate = round(statistics.mean(all_rates), 4) if all_rates else 0.0
    n_flagged = sum(1 for r in all_rates if r > 0.3)

    print(f"\n[Hallucination] Macro rate: {macro_rate:.4f}  |  Flagged: {n_flagged}/{len(all_rates)}")

    out_path = eval_file.replace("eval_", "halluc_")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "macro_hallucination_rate": macro_rate,
                "n_flagged": n_flagged,
                "n_total": len(annotated),
                "records": annotated,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[Saved] {out_path}")


# ================================
# FAILURE CASES
# ================================

def print_failure_cases(halluc_file: str, top_n: int = 5):

    with open(halluc_file, encoding="utf-8") as f:
        data = json.load(f)

    flagged = [
        r for r in data["records"]
        if r["hallucination"]["hallucination_rate"] > 0.2
    ]

    flagged.sort(key=lambda r: -r["hallucination"]["hallucination_rate"])

    print("\n" + "=" * 60)
    print("FAILURE CASES (>20% hallucination)")
    print("=" * 60)

    for rec in flagged[:top_n]:
        h = rec["hallucination"]

        print(f"\nRecord: {rec['record_id']} | {rec['category']}")
        print(f"Hallucination rate: {h['hallucination_rate']:.2%}")

        for s in h["sentences"]:
            if s["label"] == "hallucinated":
                print(f"❌ {s['sentence']}")

        print(f"Reference: {rec['reference'][:150]}...")


# ================================
# MAIN
# ================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_file", default="results/eval_rag.json")
    parser.add_argument("--rag_context_file", default=None)
    parser.add_argument("--show_failures", action="store_true")

    args = parser.parse_args()

    run_hallucination_eval(args.eval_file, args.rag_context_file)

    if args.show_failures:
        halluc_file = args.eval_file.replace("eval_", "halluc_")
        print_failure_cases(halluc_file)