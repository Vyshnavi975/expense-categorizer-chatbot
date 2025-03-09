"""Unit tests for aggregation logic and demo-mode question answering in
expenses.chat. No API key required -- these only exercise ExpenseChat's
pure aggregation methods and its pattern-matching answer() method."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from expenses.categorizer import categorize_dataframe
from expenses.chat import ExpenseChat


@pytest.fixture
def chat():
    df = pd.DataFrame(
        {
            "date": [
                "2026-06-01", "2026-06-05", "2026-06-10",
                "2026-07-01", "2026-07-15", "2026-07-20",
                "2026-08-01", "2026-08-02",
            ],
            "description": [
                "Trader Joe's #112", "Starbucks Coffee", "Shell Oil 4471",
                "Chipotle Mexican Grill", "Amazon.com Order", "Netflix.com Subscription",
                "Whole Foods Market", "Starbucks Coffee",
            ],
            "amount": [60.0, 5.0, 40.0, 15.0, 100.0, 15.0, 80.0, 5.0],
        }
    )
    categorized = categorize_dataframe(df)
    return ExpenseChat(categorized)


def test_total_spending(chat):
    assert chat.total_spending() == pytest.approx(320.0)


def test_transaction_count(chat):
    assert chat.transaction_count() == 8


def test_spending_by_category(chat):
    by_cat = chat.spending_by_category()
    assert by_cat["Groceries"] == pytest.approx(140.0)  # Trader Joe's + Whole Foods
    assert by_cat["Dining"] == pytest.approx(25.0)  # Starbucks x2 + Chipotle
    assert by_cat["Transport"] == pytest.approx(40.0)
    assert by_cat["Shopping"] == pytest.approx(100.0)
    assert by_cat["Entertainment"] == pytest.approx(15.0)


def test_top_category(chat):
    category, amount = chat.top_category()
    assert category == "Groceries"
    assert amount == pytest.approx(140.0)


def test_spending_in_month(chat):
    assert chat.spending_in_month(2026, 6) == pytest.approx(105.0)
    assert chat.spending_in_month(2026, 7) == pytest.approx(130.0)
    assert chat.spending_in_month(2026, 8) == pytest.approx(85.0)


def test_spending_for_category(chat):
    assert chat.spending_for_category("Dining") == pytest.approx(25.0)
    assert chat.spending_for_category("dining") == pytest.approx(25.0)  # case-insensitive


def test_spending_for_category_with_month_filter(chat):
    # Starbucks appears once in June and once in August
    assert chat.spending_for_category("Dining", year=2026, month=6) == pytest.approx(5.0)
    assert chat.spending_for_category("Dining", year=2026, month=8) == pytest.approx(5.0)


def test_average_transaction(chat):
    assert chat.average_transaction() == pytest.approx(40.0)


def test_most_expensive_transaction(chat):
    tx = chat.most_expensive_transaction()
    assert tx["description"] == "Amazon.com Order"
    assert tx["amount"] == pytest.approx(100.0)


def test_latest_month(chat):
    latest = chat.latest_month()
    assert (latest.year, latest.month) == (2026, 8)


# ---------------------------------------------------------------------
# Demo-mode (pattern matching) answer() tests
# ---------------------------------------------------------------------

def test_answer_top_category(chat):
    resp = chat.answer("what's my biggest category?")
    assert "Groceries" in resp
    assert "140.00" in resp


def test_answer_category_total(chat):
    resp = chat.answer("how much did I spend on dining")
    assert "$25.00" in resp


def test_answer_category_and_month(chat):
    resp = chat.answer("how much did I spend on groceries in June 2026")
    assert "$60.00" in resp


def test_answer_total_spending(chat):
    resp = chat.answer("what is my total spending")
    assert "$320.00" in resp


def test_answer_average_transaction(chat):
    resp = chat.answer("what's my average transaction")
    assert "$40.00" in resp


def test_answer_most_expensive(chat):
    resp = chat.answer("what was my most expensive transaction")
    assert "Amazon.com Order" in resp
    assert "$100.00" in resp


def test_answer_unrecognized_question_gives_helpful_fallback(chat):
    resp = chat.answer("what is the meaning of life")
    assert "demo mode" in resp.lower()
