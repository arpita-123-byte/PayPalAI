# PolicyPal AI 🏛️
### Explaining Indian Government Policies in Plain Language

A Retrieval-Augmented Generation (RAG) system fine-tuned with QLoRA on India Code PDFs. Translates complex Acts, GST circulars, and welfare scheme guidelines into clear citizen-friendly summaries — while grounding every answer in official sources to minimise hallucinations.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│       FastAPI Backend            │
│  POST /query  |  POST /baseline │
└──────────────┬──────────────────┘
               │
   ┌───────────▼───────────┐
   │   Policy Retriever     │
   │ SentenceTransformers   │
   │ → ChromaDB (cosine)   │
   └───────────┬───────────┘
               │  Top-K chunks
   ┌───────────▼───────────┐
   │   Context Assembler   │
   └───────────┬───────────┘
               │
   ┌───────────▼───────────┐
   │  Fine-tuned LLM        │
   │  LLaMA-3-8B + QLoRA   │
   │  (or Claude API)       │
   └───────────┬───────────┘
               │
   ┌───────────▼───────────┐
   │  Answer + Citations   │
   └───────────────────────┘
```

---

## Directory Structure

```
policypal/
├── data/
│   ├── raw/
│   │   ├── pdfs/
│   │   │   ├── GST/              ← India Code PDFs for GST
│   │   │   ├── General/          ← Income Tax Act sections
│   │   │   ├── WelfareScheme/    ← Consumer
│   │   │   ├── Labour/           ← Essential Commodities
│   │   │          
│   │   └── summaries.json        ← {record_id: plain-English summary}
│   ├── processed/
│   │   ├── all_records.json
│   │   └── chunks.json           ← Sample of first 500 chunks
│   ├── splits/
│   │   ├── train.json            ← 80%
│   │   ├── val.json              ← 10%
│   │   └── test.json             ← 10%
│   └── vectorstore/              ← ChromaDB persistent store
├── src/
│   ├── data_pipeline.py          ← PDF extraction, cleaning, chunking, splitting
│   ├── fine_tune.py              ← QLoRA training with PEFT + TRL
│   ├── rag_pipeline.py           ← Retrieval + generation
│   ├── evaluate.py               ← ROUGE, BLEU, readability metrics
│   ├── hallucination_check.py    ← NLI-based factual consistency check
│   └── app.py                    ← FastAPI backend
├── frontend/
│   └── index.html                ← Vanilla JS UI (no framework)
├── notebooks/
│   ├── 01_eda.ipynb              ← Dataset EDA
│   ├── 02_finetuning.ipynb       ← Training analysis
│   └── 03_evaluation.ipynb       ← ROUGE/BLEU + error analysis
├── models/                       ← Created after training
│   └── policypal-qlora/
│       └── final/                ← LoRA adapter checkpoint
├── results/                      ← Created after evaluation
│   ├── eval_baseline.json
│   ├── eval_rag.json
│   ├── eval_finetuned.json
│   ├── halluc_baseline.json
│   └── halluc_rag.json
├── .env.example                  ← Copy to .env and fill
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY (and HUGGINGFACE_TOKEN for LLaMA-3)
```

### 3. Place PDFs

```
data/raw/pdfs/
  GST/           ← cgst_act_2017.pdf, gst_circular_001.pdf, ...
  IncomeTax/     ← income_tax_act_section_80c.pdf, ...
  WelfareScheme/ ← pm_kisan_guidelines.pdf, pmay_scheme.pdf, ...
  Labour/        ← code_on_wages_2019.pdf, epf_act.pdf, ...
  Startup/       ← startup_india_action_plan.pdf, ...
```

### 4. Run data pipeline

```bash
# Extract PDFs + generate summaries via Claude API
python src/data_pipeline.py \
    --pdf_dir data/raw/pdfs \
    --summary_file data/raw/summaries.json \
    --generate_summaries

# Or — if you have manual summaries in summaries.json — skip --generate_summaries
python src/data_pipeline.py --pdf_dir data/raw/pdfs
```

### 5. Fine-tune (requires GPU ≥ 16 GB VRAM)

```bash
python src/fine_tune.py \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --train_file data/splits/train.json \
    --val_file   data/splits/val.json \
    --output_dir models/policypal-qlora \
    --epochs 3 --batch_size 4
```

### 6. Evaluate all models

```bash
python src/evaluate.py --test_file data/splits/test.json --model_type baseline
python src/evaluate.py --test_file data/splits/test.json --model_type rag
python src/evaluate.py --test_file data/splits/test.json --model_type finetuned

# Hallucination check
python src/hallucination_check.py --eval_file results/eval_rag.json --show_failures
```

### 7. Start backend

```bash
uvicorn src.app:app --reload --port 8000
```

### 8. Open UI

Open `frontend/index.html` in a browser (served locally or via any static server).

---

## Evaluation Metrics (Expected)

| Metric | Baseline | RAG | Fine-tuned |
|--------|----------|-----|------------|
| ROUGE-1 | 0.31 | 0.48 | 0.56 |
| ROUGE-2 | 0.15 | 0.27 | 0.34 |
| ROUGE-L | 0.28 | 0.43 | 0.51 |
| BLEU | 0.12 | 0.23 | 0.30 |
| Flesch Reading Ease | 51 | 63 | 68 |
| Hallucination Rate | 31% | 14% | 9% |

*Values are indicative; actual results depend on your PDF corpus size.*

---

## Model Choice — QLoRA Justification

| Approach | VRAM | Quality | Cost |
|---|---|---|---|
| Full fine-tune (bf16) | ~80 GB | ✅ Best | 💰💰💰 |
| LoRA (16-bit) | ~24 GB | ✅ Good | 💰💰 |
| **QLoRA (NF4)** | **~12 GB** | **✅ Good** | **💰** |
| Prompt Tuning | ~8 GB | ⚠ Weak | 💰 |

QLoRA with NF4 + double quantization achieves near-LoRA quality at half the VRAM, making it feasible on a single A100-40GB or T4-16GB GPU.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | RAG query (fine-tuned or Claude API) |
| POST | `/baseline` | Zero-shot baseline |
| GET | `/categories` | Supported policy categories |
| GET | `/metrics` | Latest evaluation results |
| GET | `/health` | Health check |

### Example request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is eligible for PM-KISAN?", "category": "WelfareScheme"}'
```

---

## Sources

- **India Code** (indiacode.nic.in) — Acts & Regulations (manually downloaded PDFs)
- **CBIC** (cbic.gov.in) — GST circulars and notifications
- **CBDT** (incometax.gov.in) — Income Tax Act & notifications
- **MyGov / PM portals** — Welfare scheme guidelines

---

## Project Requirements Coverage

| Criterion | Implementation |
|-----------|---------------|
| Dataset quality & preprocessing | `data_pipeline.py` — PDF extraction, section splitting, cleaning, 80/10/10 stratified split |
| PEFT Fine-tuning with justification | `fine_tune.py` — QLoRA (NF4, r=16), justification in docstring vs. alternatives |
| Baseline comparison | `evaluate.py` — zero-shot Claude vs. RAG vs. fine-tuned |
| Data storage | ChromaDB (vector) + JSON splits (structured) |
| Quantitative evaluation | ROUGE-1/2/L, BLEU, Flesch RE, hallucination rate |
| Qualitative / error analysis | `03_evaluation.ipynb` — failure cases, category breakdown, NLI flagging |
| Improvement demonstration | Metric tables + improvement % in notebook conclusions |
| Frontend UI | `frontend/index.html` — live query, comparison panel, metrics dashboard |
