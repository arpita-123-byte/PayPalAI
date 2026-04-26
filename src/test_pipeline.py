"""
test_pipeline.py
PolicyPal AI — Unit tests for data pipeline and PDF ingestor utilities.

Run:
    pytest tests/test_pipeline.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_pipeline import (
    clean_text,
    validate_record,
    chunk_record,
    split_dataset,
    PolicyRecord,
)
from pdf_ingestor import (
    strip_boilerplate,
    strip_page_numbers,
    normalise_whitespace,
    clean_india_code_text,
    infer_doc_type,
    split_into_sections,
)


# ─────────────────────────────────────────────
#  TEXT CLEANING
# ─────────────────────────────────────────────

class TestTextCleaning:
    def test_strip_page_number_pattern(self):
        text = "Some legal text.\nPage 3 of 17\nMore text."
        result = clean_text(text)
        assert "Page 3 of 17" not in result

    def test_strip_standalone_page_number(self):
        text = "Legal provision text.\n  12  \nNext section."
        result = strip_page_numbers(text)
        assert "  12  " not in result

    def test_collapse_blank_lines(self):
        text = "Line one.\n\n\n\n\nLine two."
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_remove_decorative_dashes(self):
        text = "Header\n--------------------\nContent"
        result = clean_text(text)
        assert "----" not in result

    def test_strip_india_code_boilerplate(self):
        text = "Ministry of Law and Justice\nActual content here.\n"
        result = strip_boilerplate(text)
        assert "Ministry of Law and Justice" not in result

    def test_strip_india_code_url(self):
        text = "www.indiacode.nic.in/handle/123456789\nActual text."
        result = strip_boilerplate(text)
        assert "indiacode.nic.in" not in result

    def test_normalise_whitespace(self):
        text = "Word1   \t\t   Word2"
        result = normalise_whitespace(text)
        assert "\t" not in result

    def test_clean_preserves_content(self):
        text = "The GST Composition Scheme allows small businesses to pay a flat tax rate."
        result = clean_india_code_text(text)
        assert "GST Composition Scheme" in result
        assert "flat tax rate" in result


# ─────────────────────────────────────────────
#  RECORD VALIDATION
# ─────────────────────────────────────────────

class TestValidation:
    def make_record(self, raw_words=100, summary_words=20, category="GST"):
        return PolicyRecord(
            id="test_001",
            source="IndiaCode",
            category=category,
            title="Test Policy",
            raw_text=" ".join(["word"] * raw_words),
            simple_summary=" ".join(["simple"] * summary_words),
        )

    def test_valid_record_passes(self):
        rec = self.make_record()
        assert validate_record(rec) is True

    def test_short_raw_text_fails(self):
        rec = self.make_record(raw_words=20)
        assert validate_record(rec) is False

    def test_short_summary_fails(self):
        rec = self.make_record(summary_words=5)
        assert validate_record(rec) is False

    def test_invalid_category_fails(self):
        rec = self.make_record(category="Fake")
        assert validate_record(rec) is False

    def test_all_valid_categories_pass(self):
        for cat in ["GST", "IncomeTax", "WelfareScheme", "Labour", "Startup", "General"]:
            rec = self.make_record(category=cat)
            assert validate_record(rec) is True, f"Category {cat} should be valid"


# ─────────────────────────────────────────────
#  CHUNKING
# ─────────────────────────────────────────────

class TestChunking:
    def make_record(self):
        return PolicyRecord(
            id="chunk_test",
            source="IndiaCode",
            category="GST",
            title="Chunking Test",
            raw_text=" ".join([
                "The Goods and Services Tax Act provides for levy and collection of tax."
            ] * 30),  # ~240 words → 1 or 2 chunks
            simple_summary="This is about GST levy and collection.",
        )

    def test_chunks_produced(self):
        rec   = self.make_record()
        chunks = chunk_record(rec)
        assert len(chunks) >= 1

    def test_chunk_has_required_fields(self):
        chunks = chunk_record(self.make_record())
        for c in chunks:
            assert "chunk_id"  in c
            assert "record_id" in c
            assert "text"      in c
            assert "category"  in c
            assert "title"     in c

    def test_chunk_id_unique(self):
        chunks = chunk_record(self.make_record())
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_text_nonempty(self):
        chunks = chunk_record(self.make_record())
        for c in chunks:
            assert len(c["text"].strip()) > 0


# ─────────────────────────────────────────────
#  DATASET SPLITTING
# ─────────────────────────────────────────────

class TestSplitting:
    def make_records(self, n_per_cat=10):
        records = []
        for cat in ["GST", "IncomeTax", "WelfareScheme"]:
            for i in range(n_per_cat):
                records.append(PolicyRecord(
                    id=f"{cat.lower()}_{i:03d}",
                    source="IndiaCode",
                    category=cat,
                    title=f"{cat} Policy {i}",
                    raw_text=" ".join(["word"] * 100),
                    simple_summary=" ".join(["summary"] * 20),
                ))
        return records

    def test_split_sizes_correct(self):
        records = self.make_records(10)
        train, val, test = split_dataset(records, ratios=(0.8, 0.1, 0.1))
        total = len(train) + len(val) + len(test)
        assert total == len(records)

    def test_no_overlap_between_splits(self):
        records = self.make_records(10)
        train, val, test = split_dataset(records)
        train_ids = {r.id for r in train}
        val_ids   = {r.id for r in val}
        test_ids  = {r.id for r in test}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_all_categories_in_train(self):
        records = self.make_records(20)
        train, val, test = split_dataset(records)
        train_cats = {r.category for r in train}
        assert "GST" in train_cats
        assert "IncomeTax" in train_cats
        assert "WelfareScheme" in train_cats


# ─────────────────────────────────────────────
#  PDF INGESTOR UTILITIES
# ─────────────────────────────────────────────

class TestPDFIngestor:
    def test_infer_doc_type_act(self, tmp_path):
        from pdf_ingestor import infer_doc_type
        p = tmp_path / "cgst_act_2017.pdf"
        p.touch()
        assert infer_doc_type(p) == "Act"

    def test_infer_doc_type_circular(self, tmp_path):
        from pdf_ingestor import infer_doc_type
        p = tmp_path / "gst_circular_2024_01.pdf"
        p.touch()
        assert infer_doc_type(p) == "Circular"

    def test_infer_doc_type_scheme(self, tmp_path):
        from pdf_ingestor import infer_doc_type
        p = tmp_path / "pm_kisan_scheme.pdf"
        p.touch()
        assert infer_doc_type(p) == "Scheme"

    def test_section_split_returns_list(self):
        text = (
            "1. Short title and commencement This Act may be called the Test Act.\n\n"
            "2. Definitions In this Act, unless the context otherwise requires.\n\n"
            "3. Levy of tax There shall be levied a tax called the goods and services tax.\n\n"
            "4. Registration Every supplier shall be liable to be registered.\n\n"
        )
        sections = split_into_sections(text)
        assert isinstance(sections, list)
        assert len(sections) >= 1

    def test_section_split_small_doc_no_split(self):
        text = "Short document with no sections. Just some policy text that is brief."
        sections = split_into_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == "Full Document"


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

class TestMetrics:
    def test_rouge_scores_are_float(self):
        from evaluate import compute_rouge
        scores = compute_rouge("the cat sat on the mat", "a cat sat on the mat")
        assert isinstance(scores["rouge1_f"], float)
        assert 0.0 <= scores["rouge1_f"] <= 1.0

    def test_bleu_score_is_float(self):
        from evaluate import compute_bleu
        score = compute_bleu("the cat sat on the mat", "a cat sat on the mat")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_identical_texts_rouge1_is_one(self):
        from evaluate import compute_rouge
        text = "The GST Composition Scheme allows small businesses."
        scores = compute_rouge(text, text)
        assert scores["rouge1_f"] == pytest.approx(1.0, abs=0.01)

    def test_readability_returns_dict(self):
        from evaluate import compute_readability
        result = compute_readability("The cat sat on the mat. It was a sunny day.")
        assert "flesch_reading_ease" in result
        assert "flesch_kincaid_grade" in result
