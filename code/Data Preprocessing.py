"""Prepare company price files and coverage/sector summary tables.

The downloaded S&P 500 index CSV is used as the expected trading-session
calendar. It is the Yahoo Finance ``^GSPC`` history produced by
``Data Collection.py``.
"""

from __future__ import annotations

import csv
from datetime import date
import math
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
COMPANY_DIR = PROJECT_DIR / "data" / "companies"
OUTPUT_DIR = PROJECT_DIR / "data" / "companies-prep"
MONTHLY_OUTPUT_DIR = PROJECT_DIR / "data" / "companies-prep-month"
COMPANY_SUMMARY_PATH = PROJECT_DIR / "data" / "company-summary-prep.csv"
SECTOR_SUMMARY_PATH = PROJECT_DIR / "data" / "sector-summary-prep.csv"
INDEX_CALENDAR = PROJECT_DIR / "data" / "sp500_daily_50y.csv"
INCLUDE_INCOMPLETE_MONTHS = True

METADATA_COLUMNS = {
    "membership",
    "inindex",
    "membershipsource",
    "ticker",
    "company",
    "sector",
    "delisted",
}


def parse_date(value: str, file_path: Path, line_number: int) -> date:
    """Parse the date component of a CSV date, including timezone suffixes."""
    date_text = value.strip()[:10]
    try:
        return date.fromisoformat(date_text)
    except ValueError as exc:
        raise ValueError(
            f"{file_path}:{line_number} has an invalid Date value: {value!r}"
        ) from exc


def read_calendar(path: Path) -> list[date]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected-trading-day calendar not found: {path}. "
            "Run Data Collection.py to download the Yahoo Finance ^GSPC series."
        )

    calendar: set[date] = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "Date" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'Date' column.")
        for line_number, row in enumerate(reader, start=2):
            calendar.add(parse_date(row["Date"] or "", path, line_number))

    if not calendar:
        raise ValueError(f"No trading dates were found in {path}.")
    return sorted(calendar)


def longest_missing_streak(
    expected_dates: list[date], observed_dates: set[date]
) -> tuple[int, int]:
    """Return max missing sessions and their inclusive calendar-day span."""
    longest_sessions = 0
    longest_calendar_days = 0
    streak_start: date | None = None
    streak_end: date | None = None
    streak_sessions = 0

    def record_streak() -> None:
        nonlocal longest_sessions, longest_calendar_days
        if streak_start is None or streak_end is None:
            return
        calendar_days = (streak_end - streak_start).days + 1
        if streak_sessions > longest_sessions:
            longest_sessions = streak_sessions
            longest_calendar_days = calendar_days

    for expected_day in expected_dates:
        if expected_day in observed_dates:
            record_streak()
            streak_start = None
            streak_end = None
            streak_sessions = 0
        else:
            if streak_start is None:
                streak_start = expected_day
            streak_end = expected_day
            streak_sessions += 1

    record_streak()
    return longest_sessions, longest_calendar_days


def write_monthly_company_file(
    source_path: Path,
    price_column: str,
    monthly_prices: dict[date, list[float]],
    monthly_observed_dates: dict[date, set[date]],
    calendar: list[date],
    output_dir: Path,
) -> None:
    expected_dates_by_month: dict[date, set[date]] = {}
    for trading_day in calendar:
        month_start = date(trading_day.year, trading_day.month, 1)
        expected_dates_by_month.setdefault(month_start, set()).add(trading_day)

    output_path = output_dir / f"{source_path.stem}-prep.csv"
    previous_average: float | None = None
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["Date", price_column, "Return", "LogReturn"],
        )
        writer.writeheader()
        for month_start in sorted(monthly_prices):
            expected_dates = expected_dates_by_month.get(month_start, set())
            observed_dates = monthly_observed_dates[month_start]
            if not INCLUDE_INCOMPLETE_MONTHS and (
                not expected_dates or not expected_dates.issubset(observed_dates)
            ):
                continue

            average_price = math.fsum(monthly_prices[month_start]) / len(
                monthly_prices[month_start]
            )
            if previous_average is None:
                monthly_return = math.nan
                log_return = math.nan
            else:
                monthly_return = (
                    average_price - previous_average
                ) / previous_average
                log_return = math.log(average_price / previous_average)

            writer.writerow(
                {
                    "Date": month_start.isoformat(),
                    price_column: average_price,
                    "Return": monthly_return,
                    "LogReturn": log_return,
                }
            )
            previous_average = average_price


def process_company_file(
    source_path: Path, calendar: list[date], output_dir: Path
) -> dict[str, str] | None:
    with source_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "Date" not in reader.fieldnames:
            raise ValueError(f"{source_path} must contain a 'Date' column.")

        source_fields = reader.fieldnames
        price_column = (
            "Adj Close" if "Adj Close" in source_fields else "Close"
        )
        if price_column not in source_fields:
            raise ValueError(
                f"{source_path} must contain either 'Adj Close' or 'Close' "
                "to calculate returns."
            )

        output_fields = [
            field
            for field in source_fields
            if field.strip().lower() not in METADATA_COLUMNS
            and field not in {"Return", "LogReturn"}
        ]
        output_fields.extend(["Return", "LogReturn"])
        output_path = output_dir / f"{source_path.stem}-prep.csv"
        observed_dates: set[date] = set()
        metadata: dict[str, str] = {}
        observation_count = 0
        previous_price: float | None = None
        monthly_prices: dict[date, list[float]] = {}
        monthly_observed_dates: dict[date, set[date]] = {}
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=output_fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            for line_number, row in enumerate(reader, start=2):
                observed_day = parse_date(
                    row.get("Date", ""), source_path, line_number
                )
                observed_dates.add(observed_day)
                observation_count += 1
                month_start = date(observed_day.year, observed_day.month, 1)
                monthly_observed_dates.setdefault(month_start, set())
                for key, value in row.items():
                    if (
                        key is not None
                        and key.strip().lower() in METADATA_COLUMNS
                    ):
                        metadata.setdefault(key.strip().lower(), value or "")

                raw_price = (row.get(price_column) or "").strip()
                try:
                    price = float(raw_price)
                except ValueError:
                    price = math.nan
                if not math.isfinite(price) or price <= 0:
                    price = math.nan

                if not math.isnan(price):
                    monthly_prices.setdefault(month_start, []).append(price)
                    monthly_observed_dates[month_start].add(observed_day)

                if previous_price is None or math.isnan(price):
                    daily_return = math.nan
                    log_return = math.nan
                else:
                    daily_return = (price - previous_price) / previous_price
                    log_return = math.log(price / previous_price)

                writer.writerow(
                    {
                        field: row.get(field, "")
                        for field in output_fields
                        if field not in {"Return", "LogReturn"}
                    }
                    | {
                        "Return": daily_return,
                        "LogReturn": log_return,
                    }
                )
                previous_price = None if math.isnan(price) else price

    if observation_count == 0:
        output_path.unlink()
        monthly_output_path = (
            MONTHLY_OUTPUT_DIR / f"{source_path.stem}-prep.csv"
        )
        if monthly_output_path.is_file():
            monthly_output_path.unlink()
        return None

    write_monthly_company_file(
        source_path,
        price_column,
        monthly_prices,
        monthly_observed_dates,
        calendar,
        MONTHLY_OUTPUT_DIR,
    )

    ticker = metadata.get("ticker", "").strip() or source_path.stem
    company = metadata.get("company", "").strip() or ticker
    sector = metadata.get("sector", "").strip() or "Unknown"
    delisted = metadata.get("delisted", "").strip()
    membership = metadata.get("membership", metadata.get("inindex", "")).strip()

    first_day = min(observed_dates)
    last_day = max(observed_dates)
    expected_dates = [
        trading_day
        for trading_day in calendar
        if first_day <= trading_day <= last_day
    ]
    expected_set = set(expected_dates)
    missing_dates = expected_set - observed_dates
    longest_sessions, longest_calendar_days = longest_missing_streak(
        expected_dates, observed_dates
    )

    return {
        "Ticker": ticker,
        "Company": company,
        "Sector": sector,
        "Delisted": delisted,
        "Membership": membership,
        "FirstAvailableTradingDay": first_day.isoformat(),
        "LastAvailableTradingDay": last_day.isoformat(),
        "NumberOfObservations": str(observation_count),
        "ExpectedTradingDays": str(len(expected_dates)),
        "MissingExpectedTradingDays": str(len(missing_dates)),
        "LongestMissingTradingDays": str(longest_sessions),
        "LongestMissingPeriodCalendarDays": str(longest_calendar_days),
    }


def write_company_summary(
    path: Path, rows: list[dict[str, str]]
) -> None:
    if not rows:
        raise ValueError("No non-empty company CSV files were processed.")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_sector_summary(path: Path, company_summary_path: Path) -> None:
    sectors: dict[str, set[str]] = {}
    with company_summary_path.open(newline="", encoding="utf-8-sig") as file:
        company_rows = list(csv.DictReader(file))

    for company_row in company_rows:
        sector = company_row["Sector"] or "Unknown"
        sectors.setdefault(sector, set()).add(company_row["Ticker"])

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Sector", "NumberOfCompanies", "Ticker", "Company"],
        )
        writer.writeheader()
        for sector, tickers in sorted(sectors.items()):
            members = sorted(
                (
                    row
                    for row in company_rows
                    if (row["Sector"] or "Unknown") == sector
                ),
                key=lambda row: (row["Company"].casefold(), row["Ticker"]),
            )
            for member in members:
                writer.writerow(
                    {
                        "Sector": sector,
                        "NumberOfCompanies": len(tickers),
                        "Ticker": member["Ticker"],
                        "Company": member["Company"],
                    }
                )


def main() -> None:
    if not COMPANY_DIR.is_dir():
        raise FileNotFoundError(f"Company CSV directory not found: {COMPANY_DIR}")

    calendar = read_calendar(INDEX_CALENDAR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MONTHLY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    company_rows: list[dict[str, str]] = []
    for source_path in sorted(COMPANY_DIR.glob("*.csv")):
        summary_row = process_company_file(source_path, calendar, OUTPUT_DIR)
        if summary_row is not None:
            company_rows.append(summary_row)

    if not company_rows:
        raise RuntimeError(f"No company CSV files were found in {COMPANY_DIR}.")

    company_summary_path = COMPANY_SUMMARY_PATH
    write_company_summary(company_summary_path, company_rows)
    write_sector_summary(SECTOR_SUMMARY_PATH, company_summary_path)
    print(
        f"Processed {len(company_rows)} companies. Prepared per-ticker files "
        f"are in {OUTPUT_DIR}, with monthly files in {MONTHLY_OUTPUT_DIR}; "
        f"summary tables are in {PROJECT_DIR / 'data'}."
    )
    print(
        f"Expected trading dates were read from Yahoo Finance ^GSPC history: "
        f"{INDEX_CALENDAR}."
    )


if __name__ == "__main__":
    main()
