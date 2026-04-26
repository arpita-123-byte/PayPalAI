"""
data_pipeline.py
PolicyPal AI — Dataset ingestion from manually downloaded India Code PDFs.

Directory layout expected:
  data/raw/pdfs/
    GST/          ← CGST Act, GST circulars, etc.
    IncomeTax/    ← Income Tax Act sections
    WelfareScheme/← PM-KISAN, PMAY, MGNREGA guidelines
    Labour/       ← Labour Code, EPF Act
    Startup/      ← Startup India, DPIIT policies

Each PDF is extracted → cleaned → chunked → split 80/10/10.
Simple summaries are generated via Claude API (or loaded from
data/raw/summaries.json if you've already annotated them manually).

Run:
  python src/data_pipeline.py --pdf_dir data/raw/pdfs \
      --summary_file data/raw/summaries.json \
      --generate_summaries          # adds --generate_summaries to call Claude API

Dependencies:
  pip install PyMuPDF langchain-text-splitters anthropic chromadb sentence-transformers tqdm
"""

import os, json, re, hashlib, random, argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
from collections import defaultdict
import unicodedata

import fitz                        # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ── Seed ──────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Paths ─────────────────────────────────────
DATA_DIR   = Path("data")
RAW_DIR    = DATA_DIR / "raw"
PDF_DIR    = RAW_DIR  / "pdfs"
PROC_DIR   = DATA_DIR / "processed"
SPLIT_DIR  = DATA_DIR / "splits"
VSTORE_DIR = DATA_DIR / "vectorstore"

for d in [PDF_DIR, PROC_DIR, SPLIT_DIR, VSTORE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Valid categories (must match fine_tune prompt) ──
CATEGORIES = {"GST", "IncomeTax", "WelfareScheme", "Labour", "Startup", "General"}


# ─────────────────────────────────────────────
#  1.  DATA SCHEMA
# ─────────────────────────────────────────────

@dataclass
class PolicyRecord:
    id:             str
    source:         str           # "IndiaCode", "CBIC", "CBDT", …
    category:       str           # one of CATEGORIES
    title:          str
    raw_text:       str           # extracted official text
    simple_summary: str           # gold simplified label
    url:            Optional[str] = None
    effective_date: Optional[str] = None
    lang:           str           = "en"
    pdf_file:       Optional[str] = None   # originating filename


# ─────────────────────────────────────────────
#  2.  PDF EXTRACTION
# ─────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF, page by page."""
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        print(f"  [PDF ERROR] {pdf_path.name}: {e}")
        return ""


def infer_category_from_path(pdf_path: Path) -> str:
    """Use parent folder name as category."""
    parent = pdf_path.parent.name
    mapping = {
        "GST":           "GST",
        "IncomeTax":     "IncomeTax",
        "WelfareScheme": "WelfareScheme",
        "Labour":        "Labour",
        # "Startup":       "Startup",
    }
    return mapping.get(parent, "General")


def infer_title_from_filename(pdf_path: Path) -> str:
    """Convert filename to a readable title."""
    stem = pdf_path.stem
    # e.g.  "cgst_act_2017_chapter3" → "Cgst Act 2017 Chapter3"
    title = re.sub(r"[_\-]+", " ", stem).strip().title()
    return title


def load_pdfs_from_directory(pdf_dir: Path) -> list[PolicyRecord]:
    """
    Walk pdf_dir recursively, extract text from every PDF,
    and build a PolicyRecord per document.
    """
    records: list[PolicyRecord] = []
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))

    if not pdf_files:
        print(f"[Pipeline] WARNING: No PDFs found under {pdf_dir}")
        print("  Expected layout:")
        print("    data/raw/pdfs/GST/cgst_act.pdf")
        print("    data/raw/pdfs/IncomeTax/it_act_section_80c.pdf")
        print("    …")
        return records

    print(f"[Pipeline] Found {len(pdf_files)} PDFs — extracting …")
    for pdf_path in tqdm(pdf_files, unit="pdf"):
        text = extract_pdf_text(pdf_path)
        if len(text.split()) < 30:
            print(f"  [SKIP] {pdf_path.name} — too little text ({len(text.split())} words)")
            continue

        rec_id = hashlib.md5(str(pdf_path).encode()).hexdigest()[:12]
        cat    = infer_category_from_path(pdf_path)
        title  = infer_title_from_filename(pdf_path)

        records.append(PolicyRecord(
            id=rec_id,
            source="IndiaCode",
            category=cat,
            title=title,
            raw_text=text,
            simple_summary="",          # filled later
            pdf_file=pdf_path.name,
        ))

    print(f"[Pipeline] Extracted {len(records)} records from PDFs.")
    return records


# ─────────────────────────────────────────────
#  3.  SUMMARY GENERATION  (Claude API)
# ─────────────────────────────────────────────

SUMMARY_SYSTEM = (
    "You are an expert at simplifying Indian legal and policy documents. "
    "Given a section of an official Act or policy, write a SHORT (3–5 sentence) "
    "plain-English summary that any citizen can understand. "
    "Include: what the rule/scheme is, who it applies to, key amounts or dates, "
    "and how to benefit from it. Do NOT copy sentences verbatim from the source."
)

def generate_summary_claude(raw_text: str, category: str, title: str) -> str:
    """Call Claude API to generate a simple summary (used when --generate_summaries flag is set)."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Truncate to ~1500 tokens of source text to stay within limits
    excerpt = " ".join(raw_text.split()[:600])

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=SUMMARY_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"Category: {category}\n"
                f"Title: {title}\n\n"
                f"Official text (excerpt):\n{excerpt}\n\n"
                "Write the simplified summary:"
            )
        }]
    )
    return msg.content[0].text.strip()


def load_or_generate_summaries(
    records: list[PolicyRecord],
    summary_file: Path,
    generate: bool = False,
) -> list[PolicyRecord]:
    """
    1. Load existing summaries from summary_file (JSON: {id: summary_text}).
    2. If generate=True, call Claude API for any record that still lacks a summary.
    3. Save updated summaries back to summary_file.
    """
    existing: dict[str, str] = {}
    if summary_file.exists():
        with open(summary_file) as f:
            existing = json.load(f)
        print(f"[Summaries] Loaded {len(existing)} existing summaries from {summary_file}")

    need_summary = [r for r in records if r.id not in existing and not r.simple_summary]

    if need_summary and generate:
        print(f"[Summaries] Generating summaries for {len(need_summary)} records via Claude API …")
        for rec in tqdm(need_summary, unit="rec"):
            try:
                summary = generate_summary_claude(rec.raw_text, rec.category, rec.title)
                existing[rec.id] = summary
            except Exception as e:
                print(f"  [API ERROR] {rec.id}: {e}")
                existing[rec.id] = ""
        # Save
        with open(summary_file, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"[Summaries] Saved to {summary_file}")

    elif need_summary and not generate:
        print(
            f"[Summaries] {len(need_summary)} records lack summaries. "
            "Pass --generate_summaries to auto-generate via Claude API, "
            "or add them manually to data/raw/summaries.json."
        )

    # Apply summaries to records
    for rec in records:
        if rec.id in existing:
            rec.simple_summary = existing[rec.id]

    return records


# ─────────────────────────────────────────────
#  4.  TEXT CLEANING
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize unicode, strip PDF artefacts, collapse whitespace.
    Targets India Code PDFs which often have:
      - Page headers/footers ("Ministry of Law and Justice")
      - Running headers ("THE CENTRAL GOODS AND SERVICES TAX ACT, 2017")
      - Page numbers ("1", "2 of 45")
      - Boilerplate footer text
    """
    text = unicodedata.normalize("NFKC", text)

    # Remove page numbers (standalone digits or "Page N of M")
    text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)          # lone page numbers

    # Remove common India Code header/footer boilerplate
    boilerplate_patterns = [
        r"Ministry of Law and Justice.*?\n",
        r"Legislative Department.*?\n",
        r"India Code.*?\n",
        r"www\.indiacode\.nic\.in.*?\n",
        r"Disclaimer.*?\n",
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # Remove decorative lines
    text = re.sub(r"[-_=*]{4,}", "", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove non-printable control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    return text.strip()


def split_into_sections(text: str, title: str) -> list[tuple[str, str]]:
    """
    Attempt to split a large Act PDF into logical sections by detecting
    section headers like "1.", "Section 2.", "CHAPTER III" etc.
    Returns list of (section_title, section_text) tuples.
    Falls back to a single entry if no sections detected.
    """
    section_pattern = re.compile(
        r"(?m)^(?:SECTION\s+\d+|Section\s+\d+|\d+\.\s+[A-Z][a-zA-Z ]{3,})",
    )
    matches = list(section_pattern.finditer(text))

    if len(matches) < 3:
        return [(title, text)]   # treat whole doc as one record

    sections = []
    for i, match in enumerate(matches):
        sec_title = match.group(0).strip()
        start     = match.start()
        end       = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_text  = text[start:end].strip()
        if len(sec_text.split()) >= 40:
            sections.append((sec_title, sec_text))

    return sections if sections else [(title, text)]


def expand_records_by_section(records: list[PolicyRecord]) -> list[PolicyRecord]:
    """
    For large PDFs (>2000 words), split into per-section records so that
    chunks are more semantically coherent.
    """
    expanded: list[PolicyRecord] = []
    for rec in records:
        word_count = len(rec.raw_text.split())
        if word_count < 2000:
            expanded.append(rec)
            continue

        sections = split_into_sections(rec.raw_text, rec.title)
        for i, (sec_title, sec_text) in enumerate(sections):
            new_id = f"{rec.id}_s{i:03d}"
            expanded.append(PolicyRecord(
                id=new_id,
                source=rec.source,
                category=rec.category,
                title=f"{rec.title} — {sec_title}",
                raw_text=sec_text,
                simple_summary=rec.simple_summary,  # shared until overwritten
                url=rec.url,
                pdf_file=rec.pdf_file,
            ))

    print(f"[Pipeline] Expanded to {len(expanded)} section-level records (from {len(records)} PDFs)")
    return expanded


# ─────────────────────────────────────────────
#  5.  VALIDATION
# ─────────────────────────────────────────────

def validate_record(rec: PolicyRecord) -> bool:
    return (
        len(rec.raw_text.split()) >= 50
        and rec.category in CATEGORIES
    )

# ─────────────────────────────────────────────
#  6.  CHUNKING  (for vector store)
# ─────────────────────────────────────────────

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ".", " "],
)

def chunk_record(rec: PolicyRecord) -> list[dict]:
    """Split raw_text into overlapping chunks with metadata for ChromaDB."""
    chunks = splitter.split_text(rec.raw_text)
    return [
        {
            "chunk_id":  f"{rec.id}_c{i:04d}",
            "record_id": rec.id,
            "source":    rec.source,
            "category":  rec.category,
            "title":     rec.title,
            "pdf_file":  rec.pdf_file or "",
            "text":      chunk,
        }
        for i, chunk in enumerate(chunks)
    ]


# ─────────────────────────────────────────────
#  7.  TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────

def split_dataset(
    records: list[PolicyRecord],
    ratios: tuple = (0.80, 0.10, 0.10),
) -> tuple[list, list, list]:
    """Stratified split by category to preserve label distribution."""
    buckets: dict[str, list] = defaultdict(list)
    for r in records:
        buckets[r.category].append(r)

    train, val, test = [], [], []
    for cat, items in buckets.items():
        random.shuffle(items)
        n       = len(items)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        train  += items[:n_train]
        val    += items[n_train: n_train + n_val]
        test   += items[n_train + n_val:]
        print(f"  [{cat}] total={n}  train={n_train}  val={int(n*ratios[1])}  test={n - n_train - int(n*ratios[1])}")

    random.shuffle(train)
    print(f"[Split] Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    return train, val, test


def save_splits(train, val, test) -> None:
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        out = SPLIT_DIR / f"{split_name}.json"

        # ✅ FIX: added encoding="utf-8"
        with open(out, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in split_data], f, indent=2, ensure_ascii=False)

        print(f"[Saved] {out}  ({len(split_data)} records)")

    # Also save a processed master file
    all_records = train + val + test
    master_out  = PROC_DIR / "all_records.json"

    # ✅ FIX: added encoding="utf-8"
    with open(master_out, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_records], f, indent=2, ensure_ascii=False)

    print(f"[Saved] {master_out}  ({len(all_records)} total)")


# ─────────────────────────────────────────────
#  8.  VECTOR STORE  (ChromaDB + SentenceTransformers)
# ─────────────────────────────────────────────

def build_vector_store(
    chunks: list[dict],
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    vstore_dir:  str = str(VSTORE_DIR),
) -> None:
    """Embed all chunks and upsert into a persistent ChromaDB collection."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"[VectorStore] Loading embedding model: {embed_model}")
    model  = SentenceTransformer(embed_model)
    client = chromadb.PersistentClient(path=vstore_dir)

    # Drop + recreate collection for a clean build
    try:
        client.delete_collection("policy_chunks")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="policy_chunks",
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 64
    print(f"[VectorStore] Indexing {len(chunks)} chunks in batches of {batch_size} …")
    for i in tqdm(range(0, len(chunks), batch_size), unit="batch"):
        batch      = chunks[i: i + batch_size]
        texts      = [c["text"]     for c in batch]
        ids        = [c["chunk_id"] for c in batch]
        metas      = [{k: v for k, v in c.items() if k not in ("text", "chunk_id")} for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.upsert(documents=texts, embeddings=embeddings, ids=ids, metadatas=metas)

    print(f"[VectorStore] ✓ Indexed {len(chunks)} chunks → {vstore_dir}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="PolicyPal Data Pipeline")
    parser.add_argument("--pdf_dir",            default="data/raw/pdfs",
                        help="Root directory of India Code PDFs (sub-folders = categories)")
    parser.add_argument("--summary_file",       default="data/raw/summaries.json",
                        help="JSON file mapping record_id → simple summary text")
    parser.add_argument("--generate_summaries", action="store_true",
                        help="Auto-generate missing summaries via Claude API (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--skip_vectorstore",   action="store_true",
                        help="Skip ChromaDB indexing (useful for quick dry runs)")
    parser.add_argument("--embed_model",        default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\n" + "="*60)
    print("  PolicyPal AI — Data Pipeline")
    print("="*60)

    # ── Step 1: Load PDFs ──────────────────────────────────────
    print("\n[Step 1] Loading PDFs from:", args.pdf_dir)
    raw_records = load_pdfs_from_directory(Path(args.pdf_dir))

    if not raw_records:
        print("\n[ERROR] No records loaded. Exiting.")
        raise SystemExit(1)

    # ── Step 2: Expand large PDFs into sections ────────────────
    print("\n[Step 2] Expanding large PDFs into sections …")
    raw_records = expand_records_by_section(raw_records)

    # ── Step 3: Clean text ────────────────────────────────────
    print("\n[Step 3] Cleaning text …")
    for rec in raw_records:
        rec.raw_text = clean_text(rec.raw_text)

    # ── Step 4: Load / generate summaries ─────────────────────
    # ❌ SKIPPED FOR NOW (not needed at this stage)
    # print("\n[Step 4] Loading / generating summaries …")
    # raw_records = load_or_generate_summaries(
    #     raw_records,
    #     summary_file=Path(args.summary_file),
    #     generate=args.generate_summaries,
    # )

    # ── Step 5: Validate ──────────────────────────────────────
    print("\n[Step 5] Validating records …")
    cleaned = [r for r in raw_records if validate_record(r)]
    skipped = len(raw_records) - len(cleaned)

    # ✅ FIX: removed misleading "missing summary"
    print(f"  ✓ {len(cleaned)} valid records  |  ✗ {skipped} skipped (short/invalid)")

    if not cleaned:
        print("[ERROR] No valid records after validation.")
        raise SystemExit(1)

    # ── Step 6: Split ─────────────────────────────────────────
    print("\n[Step 6] Splitting dataset …")
    train, val, test = split_dataset(cleaned)
    save_splits(train, val, test)

    # ── Step 7: Chunk ─────────────────────────────────────────
    print("\n[Step 7] Chunking records for vector store …")
    all_chunks = []
    for rec in cleaned:
        all_chunks.extend(chunk_record(rec))

    print(f"  Total chunks: {len(all_chunks)}")

    # Save chunks JSON
    chunks_out = PROC_DIR / "chunks.json"

    # ✅ FIX: added encoding="utf-8"
    with open(chunks_out, "w", encoding="utf-8") as f:
        json.dump(all_chunks[:500], f, indent=2, ensure_ascii=False)

    print(f"[Saved] {chunks_out} (first 500 chunks as sample)")

    # ── Step 8: Build vector store ────────────────────────────
    if not args.skip_vectorstore:
        print("\n[Step 8] Building ChromaDB vector store …")
        build_vector_store(all_chunks, embed_model=args.embed_model)
    else:
        print("\n[Step 8] Skipped vector store (--skip_vectorstore)")

    print("\n" + "="*60)
    print("  Pipeline complete!")
    print(f"  Records : {len(cleaned)}")
    print(f"  Chunks  : {len(all_chunks)}")
    print(f"  Splits  : data/splits/{{train,val,test}}.json")
    print(f"  VStore  : data/vectorstore/")
    print("="*60 + "\n")
