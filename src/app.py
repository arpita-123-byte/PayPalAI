"""
app.py
PolicyPal AI — FastAPI backend.

Endpoints:
  POST /query          → RAG query (fine-tuned or Claude API)
  POST /baseline       → Zero-shot baseline (no RAG, no fine-tuning)
  GET  /categories     → List supported policy categories
  GET  /health         → Health check
  GET  /metrics        → Latest evaluation metrics (if results/ exists)

Run:
  uvicorn src.app:app --reload --port 8000

Dependencies:
  pip install fastapi uvicorn pydantic python-dotenv
"""

import os, json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="PolicyPal AI",
    description="Explains Indian government policies in plain language using RAG + fine-tuned LLM.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

CATEGORIES = ["GST", "IncomeTax", "WelfareScheme", "Labour", "Startup", "General"]

# ── Lazy-load RAG pipeline to avoid cold-start at import ──
_rag = None
_baseline = None

def get_rag():
    global _rag
    if _rag is None:
        from src.rag_pipeline import PolicyPalRAG
        use_ft = os.getenv("USE_FINETUNED", "false").lower() == "true"
        _rag = PolicyPalRAG(use_finetuned=use_ft)
    return _rag

def get_baseline():
    global _baseline
    if _baseline is None:
        from src.rag_pipeline import BaselineGenerator
        _baseline = BaselineGenerator()
    return _baseline


# ─────────────────────────────────────────────
#  REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    category: Optional[str] = None   # optional filter
    top_k: int = 5

class SourceChunk(BaseModel):
    title: str
    source: str
    category: str
    score: float
    snippet: str   # first 200 chars of chunk

class QueryResponse(BaseModel):
    answer: str
    model_used: str
    sources: list[SourceChunk]

class BaselineRequest(BaseModel):
    question: str

class BaselineResponse(BaseModel):
    answer: str
    model_used: str = "baseline-claude-api"

class MetricsResponse(BaseModel):
    available: bool
    baseline: Optional[dict] = None
    rag: Optional[dict] = None
    finetuned: Optional[dict] = None


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/categories")
def categories():
    return {"categories": CATEGORIES}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if req.category and req.category not in CATEGORIES:
        raise HTTPException(400, f"Unknown category. Choose from {CATEGORIES}")

    rag = get_rag()
    try:
        resp = rag.query(req.question, top_k=req.top_k, category_filter=req.category)
    except Exception as e:
        raise HTTPException(500, f"RAG pipeline error: {e}")

    sources = [
        SourceChunk(
            title=c.title,
            source=c.source,
            category=c.category,
            score=round(c.score, 4),
            snippet=c.text[:200],
        )
        for c in resp.chunks
    ]

    return QueryResponse(answer=resp.answer, model_used=resp.model_used, sources=sources)


@app.post("/baseline", response_model=BaselineResponse)
def baseline_endpoint(req: BaselineRequest):
    gen = get_baseline()
    try:
        answer = gen.generate(req.question)
    except Exception as e:
        raise HTTPException(500, f"Baseline error: {e}")
    return BaselineResponse(answer=answer)


@app.get("/metrics", response_model=MetricsResponse)
def metrics_endpoint():
    results_dir = Path("results")
    if not results_dir.exists():
        return MetricsResponse(available=False)

    def load_agg(fname):
        p = results_dir / fname
        if p.exists():
            with open(p) as f:
                return json.load(f).get("aggregate")
        return None

    return MetricsResponse(
        available=True,
        baseline=load_agg("eval_baseline.json"),
        rag=load_agg("eval_rag.json"),
        finetuned=load_agg("eval_finetuned.json"),
    )


# ─────────────────────────────────────────────
#  MAIN (dev server)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
