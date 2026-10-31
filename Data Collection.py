"""Download up to 50 years of daily S&P 500 index history to a local CSV."""

from __future__ import annotations

import argparse
import csv
from html.parser import HTMLParser
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict
from urllib.request import Request, urlopen

import yfinance as yf


class Constituent(TypedDict):
    company: str
    delisted: bool
    sector: str
    intervals: list[tuple[date, date | None]]
    membership_complete: bool
    membership_source: str


class TableParser(HTMLParser):
    """Extract table rows from the S&P 500 constituents page."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._cell is not None:
            value = re.sub(r"\[[^\]]+\]", "", "".join(self._cell)).strip()
            self._row.append(value)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def fetch_current_constituents(as_of: date) -> dict[str, Constituent]:
    """Fetch current names and sectors; historical membership stays unknown."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Stock-Market-Network-Analysis/1.0 "
                "(academic S&P 500 data collection)"
            )
        },
    )
    with urlopen(request, timeout=30) as response:
        page = response.read().decode("utf-8")

    parser = TableParser()
    parser.feed(page)
    header_index = None
    column_indices: dict[str, int] = {}
    for row_index, row in enumerate(parser.rows):
        headers = [header.strip().lower() for header in row]
        if {"symbol", "security", "gics sector"}.issubset(headers):
            header_index = row_index
            column_indices = {
                name: headers.index(name)
                for name in ("symbol", "security", "gics sector")
            }
            break
    if header_index is None:
        raise RuntimeError(
            "Could not find the Symbol, Security, and GICS Sector columns "
            "on the current S&P 500 constituents page."
        )

    constituents: dict[str, Constituent] = {}
    for row in parser.rows[header_index + 1 :]:
        if len(row) <= max(column_indices.values()):
            continue
        symbol = row[column_indices["symbol"]].strip().replace(".", "-")
        company = row[column_indices["security"]].strip()
        sector = row[column_indices["gics sector"]].strip()
        if not symbol or not company or not sector:
            continue
        if not re.fullmatch(r"[A-Za-z0-9.^=-]+", symbol):
            continue
        constituents[symbol] = {
            "company": company,
            "delisted": False,
            "sector": sector,
            "intervals": [(as_of, None)],
            "membership_complete": False,
            "membership_source": "current constituent list only",
        }

    if not constituents:
        raise RuntimeError(
            "The current S&P 500 constituents page contained no usable "
            "company rows."
        )
    return constituents


def load_constituents(path: Path) -> dict[str, Constituent]:
    """Load ticker metadata and historical index-membership intervals."""
    required_columns = {
        "symbol",
        "company",
        "delisted",
        "membership_start",
        "membership_end",
    }
    constituents: dict[str, Constituent] = {}

    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must have these columns: "
                f"{', '.join(sorted(required_columns))}"
            )

        for line_number, row in enumerate(reader, start=2):
            symbol = (row["symbol"] or "").strip().replace(".", "-")
            company = (row["company"] or "").strip()
            delisted_text = (row["delisted"] or "").strip().lower()
            sector = (row.get("sector") or "").strip() or "Unknown"
            start_text = (row["membership_start"] or "").strip()
            end_text = (row["membership_end"] or "").strip()

            if not symbol or not company or not start_text:
                raise ValueError(
                    f"{path}:{line_number} requires symbol, company, "
                    "and membership_start."
                )
            if not re.fullmatch(r"[A-Za-z0-9.^=-]+", symbol):
                raise ValueError(
                    f"{path}:{line_number} has an unsupported Yahoo Finance "
                    f"symbol: {symbol!r}"
                )
            if delisted_text not in {"true", "false"}:
                raise ValueError(
                    f"{path}:{line_number} delisted must be true or false."
                )

            membership_start = date.fromisoformat(start_text)
            membership_end = date.fromisoformat(end_text) if end_text else None
            if membership_end is not None and membership_end < membership_start:
                raise ValueError(
                    f"{path}:{line_number} membership_end precedes "
                    "membership_start."
                )

            delisted = delisted_text == "true"
            constituent = constituents.get(symbol)
            if constituent is None:
                constituent = {
                    "company": company,
                    "delisted": delisted,
                    "sector": sector,
                    "intervals": [],
                    "membership_complete": True,
                    "membership_source": "historical constituent manifest",
                }
                constituents[symbol] = constituent
            elif (
                constituent["company"] != company
                or constituent["delisted"] != delisted
                or constituent["sector"] != sector
            ):
                raise ValueError(
                    f"{path}:{line_number} gives inconsistent metadata for "
                    f"symbol {symbol!r}."
                )

            intervals = constituent["intervals"]
            intervals.append((membership_start, membership_end))

    if not constituents:
        raise ValueError(f"{path} contains no constituent rows.")
    return constituents


def years_ago(day: date, years: int) -> date:
    """Return the calendar date a given number of years before ``day``."""
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        # Handle February 29 when the target year is not a leap year.
        return day.replace(year=day.year - years, day=28)


def is_in_index(trading_day: date, constituent: Constituent) -> bool | None:
    if (
        not constituent["membership_complete"]
        and trading_day < constituent["intervals"][0][0]
    ):
        return None
    return any(
        member_start <= trading_day
        and (member_end is None or trading_day <= member_end)
        for member_start, member_end in constituent["intervals"]
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    default_constituents = (
        project_dir / "data" / "sp500_historical_constituents.csv"
    )
    default_output = project_dir / "data" / "sp500_daily_50y.csv"
    default_company_output = project_dir / "data" / "companies"

    parser = argparse.ArgumentParser(
        description="Download daily S&P 500 index or constituent history as CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"CSV output path (default: {default_output})",
    )
    parser.add_argument(
        "--constituents",
        type=Path,
        help=(
            "Optional historical constituents CSV to merge with the current "
            "list. Requires columns "
            "symbol,company,delisted,membership_start,membership_end."
        ),
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Download only the S&P 500 index series instead of company CSVs.",
    )
    parser.add_argument(
        "--company-output",
        type=Path,
        default=default_company_output,
        help=f"Directory for per-company CSVs (default: {default_company_output})",
    )
    args = parser.parse_args()

    today = date.today()
    start = years_ago(today, 50)
    # yfinance treats `end` as exclusive; adding a day includes today's data
    # if the provider has published it.
    end = today + timedelta(days=1)

    if not args.index:
        constituents = fetch_current_constituents(today)
        manifest_path = (
            args.constituents.expanduser()
            if args.constituents is not None
            else default_constituents
        )
        if args.constituents is not None and not manifest_path.is_file():
            raise FileNotFoundError(
                f"Historical constituent manifest not found: {manifest_path}"
            )
        if manifest_path.is_file():
            historical_constituents = load_constituents(manifest_path)
            for symbol, historical in historical_constituents.items():
                current = constituents.get(symbol)
                if current is None:
                    historical["membership_source"] = (
                        "historical constituent manifest"
                    )
                    constituents[symbol] = historical
                    continue
                if historical["delisted"]:
                    raise ValueError(
                        f"Ticker {symbol} is marked as delisted in "
                        f"{manifest_path} but is also in the current "
                        "constituent list."
                    )
                current["sector"] = (
                    historical["sector"]
                    if historical["sector"] != "Unknown"
                    else current["sector"]
                )
                current["intervals"] = (
                    historical["intervals"] + current["intervals"]
                )
                current["membership_complete"] = True
                current["membership_source"] = (
                    "current list + historical constituent manifest"
                )

        if not manifest_path.is_file():
            print(
                "No historical constituent manifest found. Current companies "
                "will be downloaded, but former and delisted constituents "
                "will be absent; historical InIndex values for current "
                "companies will be blank (unknown). Add "
                f"{default_constituents} to include that history.",
                file=sys.stderr,
            )

        output_dir = args.company_output.expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        failures: list[str] = []

        for symbol, constituent in constituents.items():
            history = yf.Ticker(symbol).history(
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=False,
                timeout=30,
            )
            if history.empty:
                failures.append(symbol)
                print(f"No history returned for {symbol}.", file=sys.stderr)
                continue

            history.insert(0, "InIndex", [
                is_in_index(trading_day.date(), constituent)
                for trading_day in history.index
            ])
            history.insert(0, "Delisted", constituent["delisted"])
            history.insert(0, "Sector", constituent["sector"])
            history.insert(0, "Company", constituent["company"])
            history.insert(0, "Ticker", symbol)
            history.insert(0, "MembershipSource", constituent["membership_source"])
            output_path = output_dir / f"{symbol}.csv"
            history.to_csv(output_path, index_label="Date")
            print(f"Saved {symbol} to {output_path.resolve()}")

        if failures:
            print(
                f"Completed with {len(failures)} ticker(s) without data: "
                f"{', '.join(failures)}. Successful CSVs were retained.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        print(
            f"Saved history for {len(constituents)} tickers "
            f"({start.isoformat()} through {today.isoformat()}) "
            f"to {output_dir.resolve()}"
        )
        return

    history = yf.Ticker("^GSPC").history(
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
        timeout=30,
    )
    if history.empty:
        raise RuntimeError(
            "The provider returned no S&P 500 history. Check connectivity, "
            "provider availability, and the requested date range."
        )

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_path, index_label="Date")

    print(
        f"Saved {len(history):,} daily rows "
        f"({history.index[0].date()} to {history.index[-1].date()}) "
        f"to {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
