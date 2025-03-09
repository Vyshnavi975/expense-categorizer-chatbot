"""
Rule-based transaction categorizer.

This is the primary, always-available categorization path: it needs no API
key, no network access, and no external model. It works by matching keywords
found in a transaction's description (or merchant name) against a set of
per-category keyword lists, defined in CATEGORY_KEYWORDS below.

Matching rules
--------------
1. The description is lowercased and stripped of punctuation before matching.
2. Each category's keyword list is checked in the order categories are
   defined in CATEGORY_KEYWORDS. The first category with a matching keyword
   wins ("first match" semantics), so if you add new keywords that could
   overlap with an earlier category, keyword specificity matters more than
   which dict entry looks "more correct".
3. A keyword matches if it appears as a whole word/phrase substring in the
   cleaned description (simple substring search, not just whole-word regex,
   so "starbucks" matches "STARBUCKS #4521").
4. If nothing matches, the transaction falls into "Other".

Customizing categories
-----------------------
To add a new category or teach the categorizer new merchants:
  1. Add or edit an entry in CATEGORY_KEYWORDS (category name -> list of
     lowercase keyword strings).
  2. Keep keywords specific enough to avoid false positives (e.g. "gas" is
     risky -- it could mean a gas station (Transport) or a gas utility bill
     (Utilities); prefer "shell", "chevron", "exxon" for Transport and
     "gas company" / "natural gas" for Utilities).
  3. Category order in the dict matters for ambiguous keywords -- categories
     defined earlier are checked first.

No code changes are needed elsewhere; chat.py and cli.py just consume
whatever category labels come out of categorize_dataframe().
"""

from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

# Ordered: earlier categories are checked first when a description could
# plausibly match more than one category's keywords.
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Groceries": [
        "grocery", "groceries", "supermarket", "safeway", "kroger",
        "trader joe", "whole foods", "wholefoods", "aldi", "publix",
        "walmart supercenter", "food lion", "market basket", "sprouts",
        "wegmans", "costco wholesale",
    ],
    "Dining": [
        "restaurant", "cafe", "coffee", "starbucks", "mcdonald", "chipotle",
        "pizza", "sushi", "diner", "bistro", "grill", "taco", "burger",
        "doordash", "uber eats", "ubereats", "grubhub", "postmates",
        "dunkin", "panera", "deli", "bakery", "bar & grill", "brewery",
        "wendy", "subway sandwiches",
    ],
    "Transport": [
        "uber", "lyft", "shell", "chevron", "exxon", "gas station",
        "conoco", "bp gas", "parking", "transit", "metro card", "amtrak",
        "airlines", "delta air", "united air", "southwest air", "car rental",
        "hertz", "enterprise rent", "toll", "dmv", "auto repair", "valero",
    ],
    "Utilities": [
        "electric", "electricity", "water bill", "gas company", "natural gas",
        "internet", "comcast", "xfinity", "at&t", "verizon", "t-mobile",
        "utility", "utilities", "sewer", "waste management", "power company",
        "spectrum", "phone bill", "broadband",
    ],
    "Entertainment": [
        "netflix", "hulu", "spotify", "disney+", "disney plus", "movie",
        "cinema", "theater", "theatre", "concert", "steam games", "playstation",
        "xbox", "hbo max", "amc theatres", "ticketmaster", "youtube premium",
        "gym membership", "bowling",
    ],
    "Shopping": [
        "amazon", "target", "best buy", "ebay", "etsy", "nike", "macy's",
        "macys", "nordstrom", "ikea", "home depot", "lowe's", "lowes",
        "apple store", "clothing", "mall", "zara", "h&m", "old navy",
        "pharmacy", "cvs", "walgreens",
    ],
}

DEFAULT_CATEGORY = "Other"


def _clean(text: str) -> str:
    """Lowercase and strip punctuation for reliable substring matching."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s&+']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def categorize_description(description: str) -> str:
    """Return the best-matching category for a single description string."""
    cleaned = _clean(description)
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in cleaned:
                return category
    return DEFAULT_CATEGORY


def categorize_dataframe(
    df: pd.DataFrame, description_col: str = "description"
) -> pd.DataFrame:
    """Return a copy of df with a new 'category' column filled in via
    rule-based keyword matching against `description_col`."""
    out = df.copy()
    out["category"] = out[description_col].apply(categorize_description)
    return out


def all_categories() -> List[str]:
    """All category labels the rule-based categorizer can produce, in
    priority order, plus the fallback 'Other' category at the end."""
    return list(CATEGORY_KEYWORDS.keys()) + [DEFAULT_CATEGORY]
