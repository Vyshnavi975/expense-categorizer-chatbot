"""
Optional LLM-backed helpers.

These functions are only used when the user passes --llm on the command
line AND has OPENAI_API_KEY set in their environment. Everything here is
best-effort: if no key is present, or the API call fails for any reason,
callers should fall back to the rule-based / pattern-matching code paths
in categorizer.py and chat.py.

Supported providers
--------------------
- OpenAI (OPENAI_API_KEY set) -- uses the `openai` package.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from .categorizer import all_categories


def get_available_provider() -> Optional[str]:
    """Return 'openai' or None depending on whether the OpenAI API key
    (and importable client library) is available."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai  # noqa: F401
            return "openai"
        except ImportError:
            pass
    return None


def _call_openai(prompt: str, system: str = "") -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def call_llm(prompt: str, system: str = "") -> str:
    """Call the OpenAI provider. Raises RuntimeError if OPENAI_API_KEY
    isn't usable."""
    provider = get_available_provider()
    if provider == "openai":
        return _call_openai(prompt, system)
    raise RuntimeError(
        "No usable LLM provider found. Set OPENAI_API_KEY and install the "
        "openai client library."
    )


def llm_categorize_batch(descriptions: List[str]) -> List[str]:
    """Categorize a batch of transaction descriptions using the LLM.
    Returns a list of category strings, one per input description, in
    the same order. Falls back to 'Other' for any row the model's JSON
    response doesn't cover."""
    categories = all_categories()
    numbered = "\n".join(f"{i}: {d}" for i, d in enumerate(descriptions))
    prompt = (
        "You are categorizing personal finance transactions.\n"
        f"Allowed categories: {', '.join(categories)}.\n"
        "For each numbered transaction description below, pick exactly one "
        "category from the allowed list. Respond with ONLY a JSON object "
        'mapping the number (as a string) to the category, e.g. '
        '{\"0\": \"Dining\", \"1\": \"Groceries\"}. No other text.\n\n'
        f"Transactions:\n{numbered}"
    )
    raw = call_llm(prompt, system="You are a precise financial data assistant.")
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError:
        # Best-effort fallback: try to find a JSON object substring.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            mapping = json.loads(raw[start:end + 1])
        else:
            raise

    results = []
    for i in range(len(descriptions)):
        cat = mapping.get(str(i), "Other")
        if cat not in categories:
            cat = "Other"
        results.append(cat)
    return results


def llm_answer_question(question: str, context: str) -> str:
    """Ask the LLM a free-form question about the user's spending, given a
    text summary `context` (e.g. category/month totals) computed from the
    categorized transactions. This lets --llm mode handle questions the
    simple pattern-matcher in chat.py can't parse."""
    prompt = (
        "You are a helpful personal finance assistant. Answer the user's "
        "question ONLY using the spending data summary below. Be concise "
        "(1-4 sentences), use dollar amounts formatted like $123.45, and "
        "say so if the data doesn't answer the question.\n\n"
        f"Spending data summary:\n{context}\n\n"
        f"User question: {question}"
    )
    return call_llm(prompt, system="You are a concise, accurate personal finance assistant.")
