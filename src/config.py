"""
config.py
PolicyPal AI — Central configuration.

All paths, model names, and hyperparameters live here.
Import this anywhere instead of hardcoding values.

Usage:
    from config import CFG
    print(CFG.pdf_root)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# load_dotenv()   # reads .env file if present


@dataclass
class _Config:
    # ── Paths ──────────────────────────────────────────────────
    project_root: Path = Path(__file__).parent.parent
    pdf_root:     Path = field(default_factory=lambda: Path("data/raw/pdfs"))
    summary_file: Path = field(default_factory=lambda: Path("data/raw/summaries.json"))
    proc_dir:     Path = field(default_factory=lambda: Path("data/processed"))
    split_dir:    Path = field(default_factory=lambda: Path("data/splits"))
    vstore_dir:   Path = field(default_factory=lambda: Path("data/vectorstore"))
    results_dir:  Path = field(default_factory=lambda: Path("results"))
    models_dir:   Path = field(default_factory=lambda: Path("models"))

    # ── Categories ─────────────────────────────────────────────
    categories: tuple = ("GST", "IncomeTax", "WelfareScheme", "Labour", "Startup", "General")

    # ── Embedding model ────────────────────────────────────────
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_batch_size: int = 64

    # ── Chunking ───────────────────────────────────────────────
    chunk_size:    int = 512
    chunk_overlap: int = 64

    # ── RAG retrieval ──────────────────────────────────────────
    top_k:         int = 5
    max_ctx_chars: int = 3000

    # ── Fine-tuning ────────────────────────────────────────────
    base_model:    str = os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    ft_output_dir: str = "models/policypal-qlora"
    ft_epochs:     int = 3
    ft_batch_size: int = 4
    ft_lr:         float = 2e-4
    lora_r:        int = 16
    lora_alpha:    int = 32
    lora_dropout:  float = 0.05
    max_seq_length: int = 1024

    # ── Data splits ────────────────────────────────────────────
    train_ratio: float = 0.80
    val_ratio:   float = 0.10
    test_ratio:  float = 0.10
    random_seed: int = 42

    # # ── API keys ───────────────────────────────────────────────
    # anthropic_api_key: str = field(
    #     default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    # )
    # hf_token: str = field(
    #     default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN", "")
    # )

    # # ── Claude model used for summary generation / baseline ────
    # claude_model: str = "claude-sonnet-4-20250514"

    # ── Serving ────────────────────────────────────────────────
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    use_finetuned: bool = os.getenv("USE_FINETUNED", "false").lower() == "true"

    # ── Validation thresholds ──────────────────────────────────
    min_raw_words:     int = 50    # minimum words in raw_text to keep a record
    min_summary_words: int = 10    # minimum words in summary

    # ── Hallucination ──────────────────────────────────────────
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    halluc_flag_threshold:    float = 0.30   # rate above which verdict = "flagged"
    halluc_warning_threshold: float = 0.10   # rate above which verdict = "warning"

    def ensure_dirs(self):
        """Create all output directories if they don't exist."""
        for d in [self.proc_dir, self.split_dir, self.vstore_dir,
                  self.results_dir, self.models_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


# Singleton instance — import this everywhere
CFG = _Config()
CFG.ensure_dirs()
