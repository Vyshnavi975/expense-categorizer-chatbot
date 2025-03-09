"""
Aggregation logic and the "demo mode" natural-language question answerer.

ExpenseChat wraps a categorized transactions DataFrame (see categorizer.py)
and exposes:

  - Pure aggregation methods (total_spending, spending_by_category, ...)
    used by both demo mode and --llm mode, and directly testable without
    any NLP/LLM involved.
  - answer(question): a simple, dependency-free pattern matcher that
    recognizes a useful set of common question shapes (totals by category,
    totals by month, top/biggest category, transaction counts, averages,
    date-range spend) and answers them using the aggregation methods above.
    This is what runs when no API key is configured -- it's intentionally
    modest in scope and clearly labeled as "demo mode" wherever it's used,
    since it does not truly understand language, only recognizable patterns.

When --llm is passed and an API key is available, cli.py instead builds a
text summary via `summary_context()` and hands both that and the raw
question to expenses.llm.llm_answer_question(), which can handle much more
open-ended phrasing.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .categorizer import all_categories

MONTH_NAMES = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTH_ABBR = {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}


@dataclass
class ParsedMonth:
    year: int
    month: int

    def label(self) -> str:
        return f"{calendar.month_name[self.month]} {self.year}"


class ExpenseChat:
    """Holds a categorized transactions DataFrame and answers questions
    about it, either via simple pattern matching (demo mode) or by
    building a context summary for an LLM."""

    def __init__(self, df: pd.DataFrame):
        if "category" not in df.columns:
            raise ValueError("DataFrame must be categorized first (missing 'category' column).")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["amount"] = df["amount"].astype(float)
        self.df = df

    # ---------------------------------------------------------------
    # Pure aggregation methods (fully unit-testable, no NLP involved)
    # ---------------------------------------------------------------

    def total_spending(self) -> float:
        return round(float(self.df["amount"].sum()), 2)

    def spending_by_category(self) -> pd.Series:
        return self.df.groupby("category")["amount"].sum().sort_values(ascending=False).round(2)

    def top_category(self) -> Optional[tuple]:
        by_cat = self.spending_by_category()
        if by_cat.empty:
            return None
        return by_cat.index[0], float(by_cat.iloc[0])

    def spending_by_month(self) -> pd.Series:
        periods = self.df["date"].dt.to_period("M")
        return self.df.groupby(periods)["amount"].sum().sort_index().round(2)

    def spending_in_month(self, year: int, month: int) -> float:
        mask = (self.df["date"].dt.year == year) & (self.df["date"].dt.month == month)
        return round(float(self.df.loc[mask, "amount"].sum()), 2)

    def spending_for_category(self, category: str, year: Optional[int] = None, month: Optional[int] = None) -> float:
        mask = self.df["category"].str.lower() == category.lower()
        if year is not None:
            mask &= self.df["date"].dt.year == year
        if month is not None:
            mask &= self.df["date"].dt.month == month
        return round(float(self.df.loc[mask, "amount"].sum()), 2)

    def transaction_count(self) -> int:
        return int(len(self.df))

    def average_transaction(self) -> float:
        if self.df.empty:
            return 0.0
        return round(float(self.df["amount"].mean()), 2)

    def most_expensive_transaction(self) -> Optional[pd.Series]:
        if self.df.empty:
            return None
        return self.df.loc[self.df["amount"].idxmax()]

    def date_range(self) -> tuple:
        return self.df["date"].min(), self.df["date"].max()

    def latest_month(self) -> ParsedMonth:
        latest = self.df["date"].max()
        return ParsedMonth(latest.year, latest.month)

    def summary_context(self) -> str:
        """Build a compact text summary of the categorized data, suitable
        as LLM context for free-form question answering."""
        lines = []
        total = self.total_spending()
        lines.append(f"Total spending across all transactions: ${total:.2f}")
        lines.append(f"Number of transactions: {self.transaction_count()}")
        start, end = self.date_range()
        lines.append(f"Date range: {start.date()} to {end.date()}")
        lines.append("Spending by category:")
        for cat, amt in self.spending_by_category().items():
            lines.append(f"  - {cat}: ${amt:.2f}")
        lines.append("Spending by month:")
        for period, amt in self.spending_by_month().items():
            lines.append(f"  - {period}: ${amt:.2f}")
        top = self.top_category()
        if top:
            lines.append(f"Top spending category: {top[0]} (${top[1]:.2f})")
        most_exp = self.most_expensive_transaction()
        if most_exp is not None:
            lines.append(
                f"Most expensive single transaction: {most_exp['description']} "
                f"on {most_exp['date'].date()} for ${most_exp['amount']:.2f} "
                f"({most_exp['category']})"
            )
        return "\n".join(lines)

    # ---------------------------------------------------------------
    # Demo-mode pattern matching (no API key required)
    # ---------------------------------------------------------------

    def _find_category(self, text: str) -> Optional[str]:
        text_low = text.lower()
        for cat in all_categories():
            if cat.lower() in text_low:
                return cat
        return None

    def _find_month(self, text: str) -> Optional[ParsedMonth]:
        text_low = text.lower()

        if "last month" in text_low:
            latest = self.latest_month()
            m, y = latest.month - 1, latest.year
            if m == 0:
                m, y = 12, y - 1
            return ParsedMonth(y, m)

        if "this month" in text_low or "current month" in text_low:
            return self.latest_month()

        # "September 2025", "Sept 2025", "sep 2025"
        m = re.search(r"([a-zA-Z]+)\.?\s+(\d{4})", text_low)
        if m:
            name, year = m.group(1), int(m.group(2))
            month_num = MONTH_NAMES.get(name) or MONTH_ABBR.get(name[:3])
            if month_num:
                return ParsedMonth(year, month_num)

        # bare month name, assume the most recent matching year in the data
        for name, num in MONTH_NAMES.items():
            if re.search(rf"\b{name}\b", text_low):
                years = sorted(self.df["date"].dt.year.unique(), reverse=True)
                for y in years:
                    if ((self.df["date"].dt.year == y) & (self.df["date"].dt.month == num)).any():
                        return ParsedMonth(y, num)
                return ParsedMonth(years[0], num) if years else None

        # "2025-09" or "09/2025"
        m = re.search(r"(\d{4})-(\d{1,2})", text_low)
        if m:
            return ParsedMonth(int(m.group(1)), int(m.group(2)))
        m = re.search(r"\b(\d{1,2})/(\d{4})\b", text_low)
        if m:
            return ParsedMonth(int(m.group(2)), int(m.group(1)))

        return None

    def answer(self, question: str) -> str:
        """Answer a question using simple keyword/pattern matching. This is
        the demo-mode path used when no LLM API key is configured (or when
        --llm was not passed). It recognizes a fixed set of question
        shapes; anything else gets a helpful fallback message."""
        q = question.strip()
        q_low = q.lower()

        category = self._find_category(q)
        month = self._find_month(q)

        # Top / biggest category
        if any(kw in q_low for kw in ["biggest category", "top category", "highest category",
                                       "most spent", "which category"]):
            top = self.top_category()
            if not top:
                return "I don't have any transactions to analyze."
            return f"Your biggest spending category is **{top[0]}**, at ${top[1]:.2f}."

        # Total overall spending
        if category is None and month is None and any(
            kw in q_low for kw in ["total spending", "total spent", "how much did i spend",
                                    "how much have i spent", "overall spending"]
        ) and "category" not in q_low:
            return f"You've spent a total of ${self.total_spending():.2f} across {self.transaction_count()} transactions."

        # Category + month combo, or category alone, or month alone
        if category and month:
            amt = self.spending_for_category(category, month.year, month.month)
            return f"You spent ${amt:.2f} on {category} in {month.label()}."
        if category:
            amt = self.spending_for_category(category)
            return f"You've spent ${amt:.2f} total on {category}."
        if month:
            amt = self.spending_in_month(month.year, month.month)
            return f"You spent ${amt:.2f} total in {month.label()}."

        # Average transaction
        if "average" in q_low and "transaction" in q_low:
            return f"Your average transaction is ${self.average_transaction():.2f}."

        # Most expensive / largest single transaction
        if any(kw in q_low for kw in ["most expensive", "largest transaction", "biggest purchase",
                                       "biggest transaction"]):
            tx = self.most_expensive_transaction()
            if tx is None:
                return "I don't have any transactions to analyze."
            return (f"Your most expensive transaction was \"{tx['description']}\" "
                    f"for ${tx['amount']:.2f} on {tx['date'].date()} ({tx['category']}).")

        # How many transactions
        if "how many transactions" in q_low or "number of transactions" in q_low:
            return f"There are {self.transaction_count()} transactions in your data."

        # List categories
        if "categories" in q_low and any(kw in q_low for kw in ["what", "list", "which", "show"]):
            by_cat = self.spending_by_category()
            parts = [f"{cat} (${amt:.2f})" for cat, amt in by_cat.items()]
            return "Your spending categories are: " + ", ".join(parts) + "."

        # Breakdown / summary
        if any(kw in q_low for kw in ["breakdown", "summary", "summarize"]):
            return self.summary_context()

        # Fallback: general total, or nudge for a supported pattern.
        return (
            "I'm running in demo mode (no ANTHROPIC_API_KEY / OPENAI_API_KEY set, "
            "or --llm not passed), so I can only answer a fixed set of question "
            "patterns. Try things like: 'how much did I spend on dining last month', "
            "'what's my biggest category', 'total spending', 'how much did I spend "
            "in September 2025', 'what's my average transaction', or 'what's my "
            "most expensive transaction'. Type 'help' to see this again, or 'summary' "
            "for a full breakdown."
        )
