import argparse, json, os
from pathlib import Path
from dataclasses import dataclass, asdict

from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import textstat

Path("results").mkdir(exist_ok=True)

# ✅ FIX: ensure tokenizer is available
nltk.download("punkt", quiet=True)


# ================================
# DATA CLASS
# ================================

@dataclass
class EvalRecord:
    record_id: str
    category: str
    reference: str
    hypothesis: str
    rouge1_f: float = 0.0
    rouge2_f: float = 0.0
    rougeL_f: float = 0.0
    bleu: float = 0.0
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0


# ================================
# METRICS
# ================================

def compute_rouge(hypothesis: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)

    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bleu(hypothesis: str, reference: str) -> float:
    try:
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())

        if len(hyp_tokens) == 0:
            return 0.0

        smoother = SmoothingFunction().method1

        score = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            smoothing_function=smoother
        )
        return round(score, 4)

    except Exception:
        return 0.0


def compute_readability(text: str) -> dict:
    try:
        return {
            "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 2),
            "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 2),
        }
    except Exception:
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
        }


# ================================
# GENERATORS
# ================================

def get_generator(model_type: str):
    if model_type == "baseline":
        from rag_pipeline import PolicyPalRAG
        rag = PolicyPalRAG(use_finetuned=False)

        return lambda rec: rag.query(
            f"Explain simply: {rec['title']}"
        ).answer

    elif model_type == "rag":
        from rag_pipeline import PolicyPalRAG
        rag = PolicyPalRAG(use_finetuned=False)

        return lambda rec: rag.query(
            f"Explain the policy: {rec['title']}"
        ).answer

    elif model_type == "finetuned":
        from rag_pipeline import PolicyPalRAG
        rag = PolicyPalRAG(use_finetuned=True)

        return lambda rec: rag.query(
            f"Explain the policy: {rec['title']}"
        ).answer

    else:
        raise ValueError("Invalid model type")


# ================================
# EVALUATION LOOP
# ================================

def evaluate(test_file: str, model_type: str):

    # ✅ FIX: UTF-8 encoding
    with open(test_file, encoding="utf-8") as f:
        test_records = json.load(f)

    generate = get_generator(model_type)
    results = []

    for rec in test_records:
        print(f"Evaluating: {rec['id']} ({rec['category']})", end=" ... ")

        try:
            hypothesis = generate(rec)
        except Exception as e:
            print(f"ERROR: {e}")
            hypothesis = ""

        reference = rec.get("simple_summary", "")

        rouge = compute_rouge(hypothesis, reference)
        bleu = compute_bleu(hypothesis, reference)
        readability = compute_readability(hypothesis)

        er = EvalRecord(
            record_id=rec["id"],
            category=rec["category"],
            reference=reference,
            hypothesis=hypothesis,
            **rouge,
            bleu=bleu,
            **readability
        )

        results.append(er)

        print(f"R1={er.rouge1_f:.3f} | BLEU={er.bleu:.3f}")

    return results


# ================================
# AGGREGATION
# ================================

def aggregate(results):
    import statistics
    from collections import defaultdict

    def mean(vals):
        return round(statistics.mean(vals), 4) if vals else 0.0

    overall = {
        "n": len(results),
        "rouge1_f": mean([r.rouge1_f for r in results]),
        "rouge2_f": mean([r.rouge2_f for r in results]),
        "rougeL_f": mean([r.rougeL_f for r in results]),
        "bleu": mean([r.bleu for r in results]),
        "flesch_reading_ease": mean([r.flesch_reading_ease for r in results]),
        "flesch_kincaid_grade": mean([r.flesch_kincaid_grade for r in results]),
    }

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    per_category = {}
    for cat, recs in by_cat.items():
        per_category[cat] = {
            "rouge1_f": mean([r.rouge1_f for r in recs]),
            "bleu": mean([r.bleu for r in recs]),
        }

    return {"overall": overall, "per_category": per_category}


def print_table(agg):
    print("\n===== RESULTS =====")

    for k, v in agg["overall"].items():
        print(f"{k}: {v}")

    print("\nPer Category:")
    for cat, stats in agg["per_category"].items():
        print(cat, stats)


# ================================
# MAIN
# ================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="data/splits/test.json")
    parser.add_argument("--model_type", default="rag",
                        choices=["baseline", "rag", "finetuned"])
    args = parser.parse_args()

    print(f"\n[Evaluate] model={args.model_type}\n")

    results = evaluate(args.test_file, args.model_type)

    agg = aggregate(results)
    print_table(agg)

    out_path = f"results/eval_{args.model_type}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"aggregate": agg, "records": [asdict(r) for r in results]},
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nSaved → {out_path}")