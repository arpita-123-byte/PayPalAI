"""
conftest.py
PolicyPal AI — Pytest shared fixtures.
"""

import json
import pytest
from pathlib import Path
from dataclasses import asdict

# ── Minimal in-memory fixtures (no real PDFs or API keys needed) ──

SAMPLE_RECORDS = [
    {
        "id": "gst_test_001",
        "source": "IndiaCode",
        "category": "GST",
        "title": "CGST Act 2017 — Section 10 Composition Levy",
        "raw_text": (
            "Section 10 of the CGST Act, 2017 provides for a Composition Levy. "
            "A registered person whose aggregate turnover in the preceding financial year "
            "did not exceed one crore and fifty lakh rupees may opt to pay tax under "
            "the Composition Scheme at a flat rate. The rate for manufacturers is one percent, "
            "for restaurant services five percent, and for other eligible persons one percent. "
            "Businesses under this scheme file one quarterly return and cannot collect GST "
            "from customers or claim input tax credit."
        ),
        "simple_summary": (
            "The GST Composition Scheme lets small businesses with annual turnover under "
            "₹1.5 crore pay a flat low tax rate (1–5%) instead of standard GST slabs. "
            "It reduces paperwork to just one quarterly return. Businesses cannot collect "
            "GST from customers or claim input tax credit under this scheme."
        ),
        "pdf_file": "cgst_act_2017.pdf",
        "lang": "en",
        "url": None,
        "effective_date": None,
    },
    {
        "id": "pmkisan_test_001",
        "source": "IndiaCode",
        "category": "WelfareScheme",
        "title": "PM-KISAN Operational Guidelines",
        "raw_text": (
            "The Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) is a Central Sector scheme "
            "to provide income support to all landholding farmers' families. Under the scheme "
            "an amount of Rs 6000 per year is released in three equal installments of Rs 2000 "
            "each every four months directly to the bank accounts of the beneficiaries. "
            "All landholding farmer families are eligible. Government employees, income-tax "
            "payers, and institutional landholders are excluded from the scheme."
        ),
        "simple_summary": (
            "PM-KISAN gives ₹6,000/year to eligible farming families in three payments of "
            "₹2,000 every four months, sent directly to their bank accounts. Any farming "
            "family that owns land can apply. Government employees, income-tax payers, and "
            "institutional landholders are not eligible."
        ),
        "pdf_file": "pm_kisan_guidelines.pdf",
        "lang": "en",
        "url": None,
        "effective_date": None,
    },
    {
        "id": "ittax_test_001",
        "source": "IndiaCode",
        "category": "IncomeTax",
        "title": "Income Tax Act — New Tax Regime Section 115BAC",
        "raw_text": (
            "Under the new tax regime, income tax is levied at nil for income up to three "
            "lakh rupees; at five percent for income between three lakh and seven lakh; at "
            "ten percent for seven lakh to ten lakh; at fifteen percent for ten lakh to twelve "
            "lakh; at twenty percent for twelve lakh to fifteen lakh; and at thirty percent "
            "for income above fifteen lakh rupees. A rebate under Section 87A is available "
            "such that no tax is payable on net taxable income up to seven lakh rupees."
        ),
        "simple_summary": (
            "New tax regime slabs: 0% up to ₹3L, 5% for ₹3–7L, 10% for ₹7–10L, "
            "15% for ₹10–12L, 20% for ₹12–15L, 30% above ₹15L. "
            "If your income is ₹7L or below, you pay zero tax due to the Section 87A rebate. "
            "Salaried employees also get a ₹75,000 standard deduction."
        ),
        "pdf_file": "income_tax_act_115bac.pdf",
        "lang": "en",
        "url": None,
        "effective_date": None,
    },
]


@pytest.fixture
def sample_records() -> list[dict]:
    return SAMPLE_RECORDS


@pytest.fixture
def sample_record() -> dict:
    return SAMPLE_RECORDS[0]


@pytest.fixture
def tmp_split_files(tmp_path) -> dict:
    """Write sample records to temporary train/val/test JSON files."""
    splits = {
        "train": SAMPLE_RECORDS[:2],
        "val":   SAMPLE_RECORDS[2:3],
        "test":  SAMPLE_RECORDS[:3],
    }
    paths = {}
    for split, records in splits.items():
        p = tmp_path / f"{split}.json"
        p.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        paths[split] = str(p)
    return paths


@pytest.fixture
def tmp_summaries_file(tmp_path) -> str:
    """Write a minimal summaries.json to a temp location."""
    summaries = {r["id"]: r["simple_summary"] for r in SAMPLE_RECORDS}
    p = tmp_path / "summaries.json"
    p.write_text(json.dumps(summaries, indent=2))
    return str(p)
