"""
Command-line entry point for the Expense Categorizer Chatbot.

Usage:
    python -m expenses.cli [csv_path] [--llm] [--ask "question"] [--show]

    csv_path       Path to a transactions CSV with columns:
                    date, description, amount (default: sample_data/transactions.csv)
    --llm          Use an LLM (OpenAI, if OPENAI_API_KEY is set) for
                    categorization and for answering free-form chat
                    questions. Falls back to the rule-based categorizer and
                    demo-mode pattern matching if no key/library is
                    available.
    --ask QUESTION Answer a single question non-interactively and exit
                    (useful for scripting/demos), instead of starting the
                    interactive chat loop.
    --show         Print the categorized transaction table and exit
                    (no chat).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .categorizer import categorize_dataframe
from .chat import ExpenseChat
from .llm import get_available_provider, llm_answer_question, llm_categorize_batch

REQUIRED_COLUMNS = {"date", "description", "amount"}


def load_transactions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(sorted(missing))}. "
            f"Expected columns: date, description, amount."
        )
    return df


def categorize(df: pd.DataFrame, use_llm: bool) -> pd.DataFrame:
    if use_llm:
        provider = get_available_provider()
        if provider is None:
            print(
                "[warning] --llm was passed but no usable API key/library was found "
                "(checked OPENAI_API_KEY). Falling back to the "
                "rule-based categorizer.\n",
                file=sys.stderr,
            )
        else:
            print(f"[info] Using {provider} for LLM-based categorization...", file=sys.stderr)
            try:
                categories = llm_categorize_batch(df["description"].astype(str).tolist())
                out = df.copy()
                out["category"] = categories
                return out
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[warning] LLM categorization failed ({exc}); falling back to the "
                    "rule-based categorizer.\n",
                    file=sys.stderr,
                )
    return categorize_dataframe(df)


def print_categorized_table(df: pd.DataFrame) -> None:
    display_df = df.copy()
    display_df["amount"] = display_df["amount"].map(lambda x: f"${x:,.2f}")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(display_df[["date", "description", "amount", "category"]].to_string(index=False))


def run_chat_loop(chat: ExpenseChat, use_llm: bool) -> None:
    provider = get_available_provider() if use_llm else None
    mode_label = f"LLM mode ({provider})" if provider else "demo mode (pattern matching)"
    print("=" * 60)
    print("Expense Categorizer Chatbot")
    print(f"Loaded {chat.transaction_count()} transactions. Running in {mode_label}.")
    print("Ask a question about your spending, or type 'quit' / 'exit' to leave.")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if question.lower() == "help":
            print(
                "Bot: Try things like: 'how much did I spend on dining last month', "
                "'what's my biggest category', 'total spending', 'how much did I "
                "spend in September 2025', 'summary'."
            )
            continue

        answer = answer_question(chat, question, provider)
        print(f"Bot: {answer}")


def answer_question(chat: ExpenseChat, question: str, provider) -> str:
    if provider:
        try:
            return llm_answer_question(question, chat.summary_context())
        except Exception as exc:  # noqa: BLE001
            print(f"[warning] LLM call failed ({exc}); falling back to demo-mode answer.", file=sys.stderr)
    return chat.answer(question)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Categorize a CSV of transactions and chat about your spending."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="sample_data/transactions.csv",
        help="Path to a CSV with columns: date, description, amount (default: sample_data/transactions.csv)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use an LLM for categorization and free-form chat, if an API key is available.",
    )
    parser.add_argument(
        "--ask",
        metavar="QUESTION",
        help="Answer a single question non-interactively and exit.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the categorized transaction table and exit.",
    )
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        df = load_transactions(args.csv_path)
    except FileNotFoundError:
        print(f"Error: file not found: {args.csv_path}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    categorized = categorize(df, use_llm=args.llm)

    if args.show:
        print_categorized_table(categorized)
        return 0

    chat = ExpenseChat(categorized)

    if args.ask:
        provider = get_available_provider() if args.llm else None
        print(answer_question(chat, args.ask, provider))
        return 0

    run_chat_loop(chat, use_llm=args.llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
