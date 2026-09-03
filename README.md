# Expense Categorizer Chatbot

A command-line tool that reads a CSV of bank/credit-card transactions,
automatically categorizes each one (Groceries, Dining, Transport, Utilities,
Entertainment, Shopping, Other), and lets you ask natural-language questions
about your spending from a simple chat interface.

Categorization works out of the box with **no API key required**, using a
from-scratch keyword/rule-based matcher. If you set `OPENAI_API_KEY` (see
Setup), you can opt into an `--llm` flag for smarter, context-aware
categorization and much more flexible free-form chat.

## Features

- **Rule-based categorizer (primary path, no API key needed).** Matches
  transaction descriptions against curated keyword lists per category
  (`expenses/categorizer.py`). Fast, deterministic, fully offline, and easy
  to extend.
- **Optional LLM mode (`--llm`).** When `OPENAI_API_KEY` is set,
  categorization and chat answers can be handed off to GPT (via the `openai`
  package). If no key/library is available, `--llm` prints a warning and
  cleanly falls back to the rule-based / demo-mode paths — the app never
  crashes for lack of a key.
- **Demo-mode chat (no API key needed).** A lightweight pattern matcher
  (`expenses/chat.py`) recognizes a useful set of common questions: totals by
  category, totals by month, "last month" / "this month", top/biggest
  category, average transaction, most expensive transaction, transaction
  counts, and full summaries.
- **Pandas-powered aggregation.** All the totals, group-bys, and month
  bucketing run through `pandas`, and are unit tested independently of any
  NLP/LLM logic.
- **Realistic sample data.** `sample_data/transactions.csv` ships with 53
  original example transactions spanning three months so the whole thing is
  demoable immediately.
- **Unit tests.** `tests/` covers the categorizer and the aggregation /
  demo-mode chat logic — no API key needed to run them.

## Project structure

```
expense-categorizer-chatbot/
├── expenses/
│   ├── __init__.py
│   ├── categorizer.py   # rule-based keyword categorizer (primary path)
│   ├── chat.py          # aggregation logic + demo-mode question answering
│   ├── llm.py           # optional OpenAI-backed categorization & chat
│   └── cli.py           # argparse entry point / interactive chat loop
├── sample_data/
│   └── transactions.csv # ~53 example transactions across 3 months
├── tests/
│   ├── test_categorizer.py
│   └── test_chat.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

Requires Python 3.9+.

```bash
cd expense-categorizer-chatbot
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

`pandas` is the only hard dependency. `openai` is optional — uncomment it in
`requirements.txt` (or `pip install openai` directly) if you plan to use
`--llm`.

To use `--llm`, set:

```bash
export OPENAI_API_KEY="sk-..."
```

## Usage

Run against the bundled sample data (default if no path is given):

```bash
python3 -m expenses.cli sample_data/transactions.csv
```

Other useful flags:

```bash
# Print the categorized transaction table and exit (no chat)
python3 -m expenses.cli sample_data/transactions.csv --show

# Answer a single question non-interactively (handy for scripting/demos)
python3 -m expenses.cli sample_data/transactions.csv --ask "what's my biggest category"

# Use your own CSV
python3 -m expenses.cli path/to/your_transactions.csv

# Opt into LLM-backed categorization + free-form chat (needs an API key)
python3 -m expenses.cli sample_data/transactions.csv --llm
```

Your CSV just needs `date`, `description`, and `amount` columns (any column
order; header names are matched case-insensitively).

### Example chat session

This is an actual transcript from running the tool against
`sample_data/transactions.csv` in demo mode (no API key set):

```
$ python3 -m expenses.cli sample_data/transactions.csv
============================================================
Expense Categorizer Chatbot
Loaded 53 transactions. Running in demo mode (pattern matching).
Ask a question about your spending, or type 'quit' / 'exit' to leave.
============================================================

You: what's my biggest category
Bot: Your biggest spending category is **Transport**, at $1158.15.

You: how much did I spend on dining last month
Bot: You spent $90.30 on Dining in July 2026.

You: how much did I spend on groceries in July 2026
Bot: You spent $223.07 on Groceries in July 2026.

You: what's my average transaction
Bot: Your average transaction is $73.18.

You: what was my most expensive transaction
Bot: Your most expensive transaction was "United Airlines Ticket" for $410.00 on 2026-08-16 (Transport).

You: summary
Bot: Total spending across all transactions: $3878.36
Number of transactions: 53
Date range: 2026-06-02 to 2026-08-30
Spending by category:
  - Transport: $1158.15
  - Utilities: $744.83
  - Shopping: $719.52
  - Groceries: $693.31
  - Entertainment: $356.95
  - Dining: $205.60
Spending by month:
  - 2026-06: $1324.48
  - 2026-07: $1102.84
  - 2026-08: $1451.04
Top spending category: Transport ($1158.15)
Most expensive single transaction: United Airlines Ticket on 2026-08-16 for $410.00 (Transport)

You: quit
Goodbye!
```

Demo-mode questions it understands include variations on: "how much did I
spend on X [in <month>] [last month]", "what's my biggest/top category",
"total spending", "how many transactions", "what's my average transaction",
"what was my most expensive transaction", and "summary" / "breakdown".
Anything outside that pattern set gets a friendly nudge back toward a
supported phrasing — that's the honest boundary of rule-based demo mode.
With `--llm` and a key configured, free-form phrasing is handled by the
model instead, using a text summary of your categorized spending as context.

## Customizing the category rules

All category keywords live in one place: `CATEGORY_KEYWORDS` at the top of
`expenses/categorizer.py`. It's an ordered dict of `category name -> list of
lowercase keyword strings`. To teach it a new merchant or add a category:

1. Add or edit an entry, e.g.:
   ```python
   "Pets": ["petco", "petsmart", "vet clinic", "chewy.com"],
   ```
2. Keep keywords specific enough to avoid false positives. For example,
   `"gas"` alone is risky — it could match a gas station (Transport) or a
   gas utility bill (Utilities). Prefer specific merchant names (`"shell"`,
   `"chevron"`) for Transport and phrases like `"gas company"` /
   `"natural gas"` for Utilities.
3. Category order matters for ambiguous keywords: categories defined
   earlier in the dict are checked first, and matching uses "first match
   wins" semantics.
4. No other code needs to change — `chat.py` and `cli.py` just consume
   whatever category labels `categorize_dataframe()` produces, and
   `all_categories()` (used by chat's category detection) reads from the
   same dict automatically.

Matching itself is a simple, punctuation-stripped, case-insensitive
substring search (see `_clean()` and `categorize_description()`) — no
external dependencies, so it stays fast and fully offline.

## Running the tests

```bash
python3 -m pytest tests/ -v
```

27 tests cover the rule-based categorizer (keyword matching, case/punctuation
handling, DataFrame categorization, the "Other" fallback) and the aggregation
+ demo-mode chat logic in `ExpenseChat` (totals, group-bys, month filtering,
top category, average/most-expensive transaction, and the pattern-matching
`answer()` method). None of them require an API key.

## Notes / limitations

- The rule-based categorizer is intentionally simple and transparent — it
  will misclassify merchants it doesn't recognize as "Other" rather than
  guess. That's a feature for auditability, and it's exactly what `--llm`
  mode is for when you want broader coverage.
- Demo-mode chat only recognizes a fixed set of question shapes described
  above; it does not do general language understanding. This is called out
  explicitly in the CLI's own output so it's never mistaken for the LLM path.
- `--llm` mode makes real API calls (and incurs the associated cost/latency)
  for both categorization and chat; if the call fails for any reason (bad
  key, no network, rate limit), the tool logs a warning and falls back to
  the rule-based/demo-mode behavior rather than crashing.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Vyshnavi.
