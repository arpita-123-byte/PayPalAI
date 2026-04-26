"""
pdf_ingestor.py
PolicyPal AI — India Code PDF Ingestor

Handles all PDF-specific logic for manually downloaded India Code PDFs:
  - Detects document type (Act / Circular / Scheme / Notification)
  - Extracts metadata from filename and first page
  - Splits multi-chapter Acts into logical section records
  - Strips India Code boilerplate headers / footers
  - Outputs clean PolicyRecord objects ready for the pipeline

Expected folder layout:
    data/raw/pdfs/
        GST/
            cgst_act_2017.pdf
            igst_act_2017.pdf
            gst_circular_2024_01.pdf
        IncomeTax/
            income_tax_act_1961_chapter4.pdf
            it_notification_2024_56.pdf
        WelfareScheme/
            pm_kisan_operational_guidelines.pdf
            pmay_gramin_scheme.pdf
            mgnrega_act_2005.pdf
        Labour/
            code_on_wages_2019.pdf
            epf_miscprovisions_act_1952.pdf
            maternity_benefit_act.pdf
        Startup/
            startup_india_action_plan.pdf
            dpiit_recognition_scheme.pdf
"""

import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

CATEGORY_MAP = {
    "GST":           "GST",
    "IncomeTax":     "IncomeTax",
    "WelfareScheme": "WelfareScheme",
    "Labour":        "Labour",
    "Startup":       "Startup",
}

# India Code boilerplate patterns to strip
BOILERPLATE_PATTERNS = [
    r"Ministry of Law and Justice[^\n]*\n",
    r"Legislative Department[^\n]*\n",
    r"India Code[^\n]*\n",
    r"www\.indiacode\.nic\.in[^\n]*\n",
    r"Disclaimer[^\n]{0,200}\n",
    r"© Government of India[^\n]*\n",
    r"Printed by the Manager[^\n]*\n",
    r"THE GAZETTE OF INDIA[^\n]*\n",
    r"EXTRAORDINARY[^\n]*\n",
    r"PART II—Section [0-9].*?\n",
    r"Registered No\. D\. L\.[^\n]*\n",
]

# Section header patterns in Indian Acts
ACT_SECTION_RE = re.compile(
    r"(?m)^(?:"
    r"\d{1,3}\s*[A-Z]\.\s+"      # "2A. "
    r"|\d{1,3}\.\s+[A-Z][A-Za-z]"  # "3. Power"
    r"|CHAPTER\s+[IVXLC]+[^\n]*"  # "CHAPTER III"
    r"|PART\s+[IVXLCA-Z]+[^\n]*"  # "PART A"
    r"|SCHEDULE\s+[IVXLC]*[^\n]*" # "SCHEDULE I"
    r")"
)

# Document type detection from filename
DOC_TYPE_KEYWORDS = {
    "circular":     "Circular",
    "notification": "Notification",
    "act":          "Act",
    "guideline":    "Guidelines",
    "scheme":       "Scheme",
    "rule":         "Rules",
    "order":        "Order",
    "amendment":    "Amendment",
}


# ─────────────────────────────────────────────
#  DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class RawPolicyDoc:
    """Intermediate representation of one extracted PDF before summarisation."""
    id:            str
    category:      str
    title:         str
    doc_type:      str          # Act | Circular | Scheme | Guidelines | …
    raw_text:      str
    pdf_file:      str
    source:        str = "IndiaCode"
    simple_summary: str = ""
    lang:          str = "en"


# ─────────────────────────────────────────────
#  PAGE-LEVEL EXTRACTION
# ─────────────────────────────────────────────

def extract_pages(pdf_path: Path) -> list[str]:
    """Return list of text strings, one per page."""
    try:
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return pages
    except Exception as e:
        print(f"  [PyMuPDF ERROR] {pdf_path.name}: {e}")
        return []


def extract_full_text(pdf_path: Path) -> str:
    pages = extract_pages(pdf_path)
    return "\n".join(pages)


# ─────────────────────────────────────────────
#  METADATA INFERENCE
# ─────────────────────────────────────────────

def infer_category(pdf_path: Path) -> str:
    return CATEGORY_MAP.get(pdf_path.parent.name, "General")


def infer_doc_type(pdf_path: Path) -> str:
    stem_lower = pdf_path.stem.lower()
    for kw, label in DOC_TYPE_KEYWORDS.items():
        if kw in stem_lower:
            return label
    return "Act"   # default for India Code


def infer_title(pdf_path: Path, first_page_text: str) -> str:
    """
    Try to extract a clean title from the first page.
    Falls back to a humanised filename if nothing clean found.
    """
    # Try first non-empty line from page 1 that looks like a title
    for line in first_page_text.split("\n"):
        line = line.strip()
        if (
            15 < len(line) < 120
            and not line.startswith("www.")
            and not re.match(r"^\d", line)
            and not "Ministry" in line
        ):
            return line

    # Fallback: humanise filename
    stem = re.sub(r"[_\-]+", " ", pdf_path.stem)
    return stem.strip().title()


def make_record_id(pdf_path: Path, section_index: int = 0) -> str:
    key = f"{pdf_path.stem}_{section_index}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────
#  TEXT CLEANING (India Code specific)
# ─────────────────────────────────────────────

def strip_boilerplate(text: str) -> str:
    """Remove India Code header/footer/margin boilerplate."""
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text


def strip_page_numbers(text: str) -> str:
    """Remove standalone page numbers and 'Page N of M' patterns."""
    text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    return text


def normalise_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{3,}", "  ", text)
    text = re.sub(r"[-_=*]{4,}", "", text)          # decorative lines
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # control chars
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    return text.strip()


def clean_india_code_text(text: str) -> str:
    text = strip_boilerplate(text)
    text = strip_page_numbers(text)
    text = normalise_whitespace(text)
    return text


# ─────────────────────────────────────────────
#  SECTION SPLITTING
# ─────────────────────────────────────────────

def split_into_sections(text: str, min_words: int = 60) -> list[tuple[str, str]]:
    """
    Split a large India Code Act into logical sections using header detection.
    Returns list of (section_heading, section_text) tuples.
    Falls back to the full document as a single section if detection fails.
    """
    matches = list(ACT_SECTION_RE.finditer(text))
    if len(matches) < 3:
        return [("Full Document", text)]

    sections = []
    for i, match in enumerate(matches):
        heading = match.group(0).strip()
        start   = match.start()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body    = text[start:end].strip()
        if len(body.split()) >= min_words:
            sections.append((heading, body))

    return sections if sections else [("Full Document", text)]


# ─────────────────────────────────────────────
#  MAIN INGESTOR CLASS
# ─────────────────────────────────────────────

class IndiaCodeIngestor:
    """
    Ingests manually downloaded India Code PDFs and returns
    a flat list of RawPolicyDoc records ready for summarisation.
    """

    def __init__(self, pdf_root: str = "data/raw/pdfs", split_sections: bool = True):
        self.pdf_root      = Path(pdf_root)
        self.split_sections = split_sections

    def ingest_all(self, verbose: bool = True) -> list[RawPolicyDoc]:
        pdf_files = sorted(self.pdf_root.rglob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDFs found under '{self.pdf_root}'.\n"
                "Expected layout:\n"
                "  data/raw/pdfs/GST/cgst_act_2017.pdf\n"
                "  data/raw/pdfs/IncomeTax/income_tax_act.pdf\n"
                "  ..."
            )

        if verbose:
            print(f"[Ingestor] Found {len(pdf_files)} PDFs under {self.pdf_root}")

        all_docs: list[RawPolicyDoc] = []
        for pdf_path in pdf_files:
            docs = self.ingest_one(pdf_path, verbose=verbose)
            all_docs.extend(docs)

        if verbose:
            print(f"[Ingestor] Total records produced: {len(all_docs)}")

        return all_docs

    def ingest_one(self, pdf_path: Path, verbose: bool = True) -> list[RawPolicyDoc]:
        """Extract, clean, and optionally section-split a single PDF."""
        pages = extract_pages(pdf_path)
        if not pages:
            return []

        full_text  = "\n".join(pages)
        first_page = pages[0] if pages else ""

        # Metadata
        category = infer_category(pdf_path)
        doc_type = infer_doc_type(pdf_path)
        title    = infer_title(pdf_path, first_page)

        # Clean
        cleaned = clean_india_code_text(full_text)

        word_count = len(cleaned.split())
        if word_count < 40:
            if verbose:
                print(f"  [SKIP] {pdf_path.name} — only {word_count} words after cleaning")
            return []

        # Section-split large Acts; keep short docs as-is
        if self.split_sections and word_count > 1500:
            sections = split_into_sections(cleaned)
        else:
            sections = [("Full Document", cleaned)]

        docs: list[RawPolicyDoc] = []
        for i, (sec_heading, sec_text) in enumerate(sections):
            sec_title = title if sec_heading == "Full Document" else f"{title} — {sec_heading}"
            docs.append(RawPolicyDoc(
                id       = make_record_id(pdf_path, i),
                category = category,
                title    = sec_title,
                doc_type = doc_type,
                raw_text = sec_text,
                pdf_file = pdf_path.name,
            ))

        if verbose:
            print(f"  {pdf_path.name:<45} → {len(docs):>3} record(s)  [{category}]")

        return docs

    def ingest_category(self, category: str, verbose: bool = True) -> list[RawPolicyDoc]:
        """Ingest only PDFs from a specific category subfolder."""
        cat_dir = self.pdf_root / category
        if not cat_dir.exists():
            raise FileNotFoundError(f"Category folder not found: {cat_dir}")
        pdf_files = sorted(cat_dir.glob("*.pdf"))
        docs = []
        for pdf_path in pdf_files:
            docs.extend(self.ingest_one(pdf_path, verbose=verbose))
        return docs


# ─────────────────────────────────────────────
#  QUICK SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ingestor = IndiaCodeIngestor(pdf_root="data/raw/pdfs", split_sections=True)
    try:
        docs = ingestor.ingest_all()
        print(f"\n── Sample record ──")
        d = docs[0]
        print(f"ID       : {d.id}")
        print(f"Category : {d.category}")
        print(f"Type     : {d.doc_type}")
        print(f"Title    : {d.title}")
        print(f"Words    : {len(d.raw_text.split())}")
        print(f"Preview  : {d.raw_text[:300]}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
