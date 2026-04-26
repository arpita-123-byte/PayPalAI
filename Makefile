# PolicyPal AI — Makefile
# Usage: make <target>
# Requires: GNU Make, Python 3.10+, pip

PYTHON   = python
SRC      = src
DATA     = data
RESULTS  = results
MODELS   = models

.PHONY: help install ingest summarise pipeline train evaluate hallucination serve clean

# ──────────────────────────────────────────────
#  HELP
# ──────────────────────────────────────────────
help:
	@echo ""
	@echo "PolicyPal AI — available targets"
	@echo "──────────────────────────────────"
	@echo "  make install          Install all Python dependencies"
	@echo "  make ingest           Test PDF ingestor on your PDFs"
	@echo "  make summarise        Generate summaries via Claude API"
	@echo "  make pipeline         Full data pipeline (extract → clean → split → index)"
	@echo "  make train            QLoRA fine-tuning (requires GPU)"
	@echo "  make evaluate         Run ROUGE/BLEU evaluation for all models"
	@echo "  make hallucination    Run NLI-based hallucination check"
	@echo "  make serve            Start FastAPI backend"
	@echo "  make notebook         Launch Jupyter notebooks"
	@echo "  make clean            Remove generated data, models, results"
	@echo ""


# ──────────────────────────────────────────────
#  INSTALL
# ──────────────────────────────────────────────
install:
	pip install -r requirements.txt
	$(PYTHON) -c "import nltk; nltk.download('punkt', quiet=True)"
	@echo "✓ Dependencies installed"


# ──────────────────────────────────────────────
#  DATA PIPELINE
# ──────────────────────────────────────────────

# Quick smoke-test of the PDF ingestor
ingest:
	$(PYTHON) $(SRC)/pdf_ingestor.py

# Generate plain-English summaries via Claude API for all records that lack one
summarise:
	$(PYTHON) $(SRC)/data_pipeline.py \
		--pdf_dir $(DATA)/raw/pdfs \
		--summary_file $(DATA)/raw/summaries.json \
		--generate_summaries \
		--skip_vectorstore

# Full pipeline: extract → clean → section-split → summarise → split → embed → index
pipeline:
	$(PYTHON) $(SRC)/data_pipeline.py \
		--pdf_dir $(DATA)/raw/pdfs \
		--summary_file $(DATA)/raw/summaries.json \
		--generate_summaries

# Run pipeline without regenerating summaries (use existing summaries.json)
pipeline-nosummary:
	$(PYTHON) $(SRC)/data_pipeline.py \
		--pdf_dir $(DATA)/raw/pdfs \
		--summary_file $(DATA)/raw/summaries.json


# ──────────────────────────────────────────────
#  TRAINING
# ──────────────────────────────────────────────
train:
	$(PYTHON) $(SRC)/fine_tune.py \
		--train_file $(DATA)/splits/train.json \
		--val_file   $(DATA)/splits/val.json \
		--output_dir $(MODELS)/policypal-qlora \
		--epochs 3 \
		--batch_size 4

# Merge LoRA adapter into base model for faster inference
merge:
	$(PYTHON) $(SRC)/fine_tune.py \
		--train_file $(DATA)/splits/train.json \
		--val_file   $(DATA)/splits/val.json \
		--output_dir $(MODELS)/policypal-qlora \
		--epochs 0 \
		--merge \
		--merge_output $(MODELS)/policypal-merged


# ──────────────────────────────────────────────
#  EVALUATION
# ──────────────────────────────────────────────
evaluate:
	@mkdir -p $(RESULTS)
	$(PYTHON) $(SRC)/evaluate.py --test_file $(DATA)/splits/test.json --model_type baseline
	$(PYTHON) $(SRC)/evaluate.py --test_file $(DATA)/splits/test.json --model_type rag
	@if [ -d "$(MODELS)/policypal-qlora/final" ]; then \
		$(PYTHON) $(SRC)/evaluate.py --test_file $(DATA)/splits/test.json --model_type finetuned; \
	else \
		echo "[skip] Fine-tuned model not found — run 'make train' first"; \
	fi
	@echo "✓ Results saved to $(RESULTS)/"

hallucination:
	@mkdir -p $(RESULTS)
	@if [ -f "$(RESULTS)/eval_baseline.json" ]; then \
		$(PYTHON) $(SRC)/hallucination_check.py \
			--eval_file $(RESULTS)/eval_baseline.json --show_failures; \
	fi
	@if [ -f "$(RESULTS)/eval_rag.json" ]; then \
		$(PYTHON) $(SRC)/hallucination_check.py \
			--eval_file $(RESULTS)/eval_rag.json --show_failures; \
	fi


# ──────────────────────────────────────────────
#  SERVE
# ──────────────────────────────────────────────
serve:
	uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 2


# ──────────────────────────────────────────────
#  NOTEBOOKS
# ──────────────────────────────────────────────
notebook:
	jupyter notebook notebooks/


# ──────────────────────────────────────────────
#  CLEAN
# ──────────────────────────────────────────────
clean-data:
	rm -rf $(DATA)/processed $(DATA)/splits $(DATA)/vectorstore
	@echo "✓ Cleaned processed data"

clean-models:
	rm -rf $(MODELS)
	@echo "✓ Cleaned models"

clean-results:
	rm -rf $(RESULTS)
	@echo "✓ Cleaned results"

clean: clean-data clean-results
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Full clean done (models kept)"
