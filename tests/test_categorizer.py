"""Unit tests for the rule-based categorizer. No API key required."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from expenses.categorizer import (
    all_categories,
    categorize_dataframe,
    categorize_description,
)


def test_groceries_keyword_match():
    assert categorize_description("Trader Joe's #112") == "Groceries"
    assert categorize_description("SAFEWAY GROCERY") == "Groceries"


def test_dining_keyword_match():
    assert categorize_description("Starbucks Coffee") == "Dining"
    assert categorize_description("Chipotle Mexican Grill") == "Dining"
    assert categorize_description("Doordash Order") == "Dining"


def test_transport_keyword_match():
    assert categorize_description("Uber Trip 8842") == "Transport"
    assert categorize_description("Shell Oil 4471") == "Transport"
    assert categorize_description("Delta Airlines Ticket") == "Transport"


def test_utilities_keyword_match():
    assert categorize_description("Comcast Xfinity Internet") == "Utilities"
    assert categorize_description("Verizon Wireless Bill") == "Utilities"
    assert categorize_description("Electric Company Payment") == "Utilities"


def test_entertainment_keyword_match():
    assert categorize_description("Netflix.com Subscription") == "Entertainment"
    assert categorize_description("AMC Theatres") == "Entertainment"
    assert categorize_description("Spotify Premium") == "Entertainment"


def test_shopping_keyword_match():
    assert categorize_description("Amazon.com Order") == "Shopping"
    assert categorize_description("Best Buy Electronics") == "Shopping"
    assert categorize_description("Target Store 221") == "Shopping"


def test_unknown_description_falls_back_to_other():
    assert categorize_description("Some Totally Unrecognizable Merchant XYZ123") == "Other"


def test_case_and_punctuation_insensitive():
    assert categorize_description("STARBUCKS #4521!!") == "Dining"
    assert categorize_description("starbucks #4521") == "Dining"


def test_categorize_dataframe_adds_category_column():
    df = pd.DataFrame(
        {
            "date": ["2026-06-01", "2026-06-02"],
            "description": ["Starbucks Coffee", "Shell Oil 4471"],
            "amount": [6.75, 38.50],
        }
    )
    out = categorize_dataframe(df)
    assert "category" in out.columns
    assert list(out["category"]) == ["Dining", "Transport"]
    # original df is untouched
    assert "category" not in df.columns


def test_all_categories_includes_other_last():
    cats = all_categories()
    assert cats[-1] == "Other"
    assert "Groceries" in cats
    assert "Dining" in cats
