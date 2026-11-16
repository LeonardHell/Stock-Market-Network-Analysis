"""Plot monthly return correlations by lag and by initial-days averaging span."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
COHORT_PATH = DATA_DIR / "Results 0.1" / "sector-grouped-analysis-cohort.csv"
MONTHLY_DATA_DIR = DATA_DIR / "companies-prep-month"
DAILY_DATA_DIR = DATA_DIR / "companies-prep"
DEFAULT_OUTPUT_DIR = DATA_DIR / "Results 0.3" / "Monthly Lagged Correlation"
DEFAULT_WINDOW_YEARS = 15
DEFAULT_MAX_LAG_MONTHS = 24
DEFAULT_MAX_AVERAGED_DAYS = 80
DEFAULT_ROLLING_WINDOW_MONTHS = 12


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def month_index(value: str, source: Path, row_number: int) -> int:
    try:
        parsed_date = date.fromisoformat(value.strip()[:10])
    except ValueError as exc:
        raise ValueError(
            f"{source}, row {row_number}: invalid date {value!r}"
        ) from exc
    return parsed_date.year * 12 + parsed_date.month - 1


def month_start(value: int) -> date:
    year, zero_based_month = divmod(value, 12)
    return date(year, zero_based_month + 1, 1)


def load_cohort(path: Path) -> list[dict[str, str]]:
    cohort = read_csv(path)
    if not cohort:
        raise ValueError(f"No companies found in cohort file: {path}")
    required = {"Ticker", "Company"}
    if not required.issubset(cohort[0]):
        raise ValueError(
            f"{path} must contain columns {', '.join(sorted(required))}."
        )

    ordered = sorted(cohort, key=lambda row: row["Ticker"].strip().casefold())
    tickers = [row["Ticker"].strip() for row in ordered]
    if any(not ticker for ticker in tickers):
        raise ValueError(f"{path} contains an empty ticker.")
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"{path} contains duplicate tickers.")
    return ordered


def load_monthly_returns(
    cohort: list[dict[str, str]], source_dir: Path
) -> dict[str, dict[int, float]]:
    returns: dict[str, dict[int, float]] = {}
    for company in cohort:
        ticker = company["Ticker"].strip()
        path = source_dir / f"{ticker}-prep.csv"
        data: dict[int, float] = {}
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not {"Date", "Return"}.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"{path} must contain Date and Return columns. "
                    'Run "Data Preprocessing.py" first.'
                )
            for row_number, row in enumerate(reader, start=2):
                month = month_index(row.get("Date", ""), path, row_number)
                try:
                    value = float(row.get("Return", ""))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    if month in data:
                        raise ValueError(
                            f"{path} contains multiple returns for "
                            f"{month_start(month).isoformat()}."
                        )
                    data[month] = value
        if not data:
            raise ValueError(f"No valid monthly returns found for {ticker} in {path}.")
        returns[ticker] = data
    return returns


def load_daily_prices(
    cohort: list[dict[str, str]], source_dir: Path
) -> dict[str, dict[date, float]]:
    prices: dict[str, dict[date, float]] = {}
    for company in cohort:
        ticker = company["Ticker"].strip()
        path = source_dir / f"{ticker}-prep.csv"
        data: dict[date, float] = {}
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or not {"Date", "Adj Close"}.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"{path} must contain Date and Adj Close columns. "
                    'Run "Data Preprocessing.py" first.'
                )
            for row_number, row in enumerate(reader, start=2):
                raw_date = row.get("Date", "").strip()
                try:
                    trading_day = date.fromisoformat(raw_date[:10])
                except ValueError as exc:
                    raise ValueError(
                        f"{path}, row {row_number}: invalid date {raw_date!r}"
                    ) from exc
                try:
                    value = float(row.get("Adj Close", ""))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0:
                    if trading_day in data:
                        raise ValueError(
                            f"{path} contains multiple prices for "
                            f"{trading_day.isoformat()}."
                        )
                    data[trading_day] = value
        if not data:
            raise ValueError(f"No valid adjusted prices found for {ticker} in {path}.")
        prices[ticker] = data
    return prices


def average_lagged_correlations(
    returns: dict[str, dict[int, float]],
    max_lag_months: int,
) -> tuple[int, int, list[dict[str, float | int]]]:
    if len(returns) < 2:
        raise ValueError("At least two companies are required.")

    available_months = set.union(
        *(set(company_returns) for company_returns in returns.values())
    )
    if not available_months:
        raise ValueError("No valid monthly returns were found.")
    window_start = min(available_months)
    window_end = max(available_months) + 1
    tickers = sorted(returns)
    results: list[dict[str, float | int]] = []

    for lag in range(max_lag_months + 1):
        correlations: list[float] = []
        observation_counts: list[int] = []
        for ticker_a in tickers:
            for ticker_b in tickers:
                if ticker_a == ticker_b:
                    continue

                paired_months = sorted(
                    month
                    for month in returns[ticker_a]
                    if month + lag in returns[ticker_b]
                )
                values_a = [returns[ticker_a][month] for month in paired_months]
                values_b = [
                    returns[ticker_b][month + lag] for month in paired_months
                ]

                if len(values_a) < 2:
                    continue
                correlation = float(np.corrcoef(values_a, values_b)[0, 1])
                if math.isfinite(correlation):
                    correlations.append(correlation)
                    observation_counts.append(len(values_a))

        if not correlations:
            raise ValueError(
                f"No valid cross-company correlations could be calculated "
                f"for lag {lag}."
            )
        results.append(
            {
                "LagMonths": lag,
                "AverageCorrelation": math.fsum(correlations)
                / len(correlations),
                "CorrelationStdDev": float(np.std(correlations, ddof=1))
                if len(correlations) > 1
                else 0.0,
                "CorrelatedCompanyPairs": len(correlations),
                "MeanMonthlyObservationsPerPair": math.fsum(observation_counts)
                / len(observation_counts),
            }
        )
    return window_start, window_end, results


def rolling_one_month_lag_correlations(
    returns: dict[str, dict[int, float]],
    window_months: int,
) -> tuple[int, int, list[dict[str, float | int]]]:
    if len(returns) < 2:
        raise ValueError("At least two companies are required.")
    if window_months < 2:
        raise ValueError("The rolling window must contain at least two months.")

    available_months = set.union(
        *(set(company_returns) for company_returns in returns.values())
    )
    if not available_months:
        raise ValueError("No valid monthly returns were found.")
    first_month = min(available_months)
    end_month = max(available_months) + 1
    tickers = sorted(returns)
    results: list[dict[str, float | int]] = []

    for window_start in range(
        first_month, end_month - window_months - 1 + 1
    ):
        origin_months = range(window_start, window_start + window_months)
        correlations: list[float] = []
        observation_counts: list[int] = []
        for ticker_a in tickers:
            for ticker_b in tickers:
                if ticker_a == ticker_b:
                    continue
                paired_months = [
                    month
                    for month in origin_months
                    if month in returns[ticker_a]
                    and month + 1 in returns[ticker_b]
                ]
                if len(paired_months) < 2:
                    continue
                values_a = [returns[ticker_a][month] for month in paired_months]
                values_b = [
                    returns[ticker_b][month + 1] for month in paired_months
                ]
                correlation = float(np.corrcoef(values_a, values_b)[0, 1])
                if math.isfinite(correlation):
                    correlations.append(correlation)
                    observation_counts.append(len(paired_months))

        if correlations:
            results.append(
                {
                    "WindowStart": window_start,
                    "AverageCorrelation": math.fsum(correlations)
                    / len(correlations),
                    "CorrelationStdDev": float(np.std(correlations, ddof=1))
                    if len(correlations) > 1
                    else 0.0,
                    "CorrelatedCompanyPairs": len(correlations),
                    "MeanMonthlyObservationsPerPair": math.fsum(
                        observation_counts
                    )
                    / len(observation_counts),
                }
            )

    if not results:
        raise ValueError(
            "No rolling one-month-lag correlations could be calculated."
        )
    return first_month, end_month, results


def average_correlations_by_averaged_days(
    prices: dict[str, dict[date, float]],
    window_start: int,
    window_years: int,
    max_averaged_days: int,
) -> list[dict[str, float | int]]:
    if len(prices) < 2:
        raise ValueError("At least two companies are required.")

    window_end = window_start + window_years * 12
    price_months = range(window_start - 1, window_end)
    tickers = sorted(prices)
    prices_by_ticker_and_month: dict[
        str, dict[int, list[tuple[date, float]]]
    ] = {}
    for ticker in tickers:
        monthly_observations: dict[int, list[tuple[date, float]]] = {}
        for trading_day, price in prices[ticker].items():
            month = trading_day.year * 12 + trading_day.month - 1
            if month in price_months:
                monthly_observations.setdefault(month, []).append(
                    (trading_day, price)
                )
        prices_by_ticker_and_month[ticker] = {
            month: sorted(observations)
            for month, observations in monthly_observations.items()
        }

    results: list[dict[str, float | int]] = []
    for averaged_days in range(1, max_averaged_days + 1):
        monthly_returns: dict[str, dict[int, float]] = {}
        for ticker in tickers:
            monthly_average_prices: dict[int, float] = {}
            for month in price_months:
                first_observations = prices_by_ticker_and_month[ticker].get(
                    month, []
                )[:averaged_days]
                if first_observations:
                    monthly_average_prices[month] = (
                        math.fsum(price for _, price in first_observations)
                        / len(first_observations)
                    )

            company_returns: dict[int, float] = {}
            for month in range(window_start, window_end):
                previous_price = monthly_average_prices.get(month - 1)
                current_price = monthly_average_prices.get(month)
                if previous_price and current_price:
                    company_returns[month] = (
                        current_price - previous_price
                    ) / previous_price
            monthly_returns[ticker] = company_returns

        correlations: list[float] = []
        observation_counts: list[int] = []
        for ticker_a in tickers:
            for ticker_b in tickers:
                if ticker_a == ticker_b:
                    continue
                common_months = sorted(
                    set(monthly_returns[ticker_a])
                    & set(monthly_returns[ticker_b])
                )
                if len(common_months) < 2:
                    continue
                values_a = [monthly_returns[ticker_a][m] for m in common_months]
                values_b = [monthly_returns[ticker_b][m] for m in common_months]
                correlation = float(np.corrcoef(values_a, values_b)[0, 1])
                if math.isfinite(correlation):
                    correlations.append(correlation)
                    observation_counts.append(len(common_months))

        if not correlations:
            raise ValueError(
                f"No valid cross-company correlations could be calculated "
                f"when averaging the first {averaged_days} trading days."
            )
        results.append(
            {
                "AveragedFirstTradingDays": averaged_days,
                "AverageCorrelation": math.fsum(correlations)
                / len(correlations),
                "CorrelatedCompanyPairs": len(correlations),
                "MeanMonthlyObservationsPerPair": math.fsum(observation_counts)
                / len(observation_counts),
            }
        )
    return results


def write_results(
    output_dir: Path,
    window_start: int,
    window_end: int,
    results: list[dict[str, float | int]],
    company_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    start_date = month_start(window_start)
    end_date = month_start(window_end)
    csv_path = output_dir / "average-lagged-monthly-return-correlations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "LagMonths",
            "AverageCorrelation",
            "CorrelationStdDev",
            "CorrelatedCompanyPairs",
            "MeanMonthlyObservationsPerPair",
            "WindowStart",
            "WindowEndExclusive",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    **row,
                    "WindowStart": start_date.isoformat(),
                    "WindowEndExclusive": end_date.isoformat(),
                }
            )

    lags = [int(row["LagMonths"]) for row in results]
    average_correlations = [
        float(row["AverageCorrelation"]) for row in results
    ]
    standard_deviations = [
        float(row["CorrelationStdDev"]) for row in results
    ]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.errorbar(
        lags,
        average_correlations,
        yerr=standard_deviations,
        marker="o",
        linewidth=1.5,
        capsize=3,
        label="Mean ± 1 standard deviation across company pairs",
    )
    axis.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axis.set(
        xlabel="Distance between monthly returns (months)",
        ylabel="Average cross-company correlation",
        title=(
            f"Lagged monthly-return correlation across {company_count} companies\n"
            f"All available data: {start_date:%Y-%m} to "
            f"{month_start(window_end - 1):%Y-%m}"
        ),
        xlim=(-0.5, len(lags) - 0.5),
        ylim=(-1, 1),
        xticks=lags,
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "average-lagged-monthly-return-correlations.png",
        dpi=180,
    )
    plt.close(figure)


def write_rolling_results(
    output_dir: Path,
    results: list[dict[str, float | int]],
    window_months: int,
    company_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rolling-one-month-lag-correlations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "WindowStart",
            "AverageCorrelation",
            "CorrelationStdDev",
            "CorrelatedCompanyPairs",
            "MeanMonthlyObservationsPerPair",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    **row,
                    "WindowStart": month_start(
                        int(row["WindowStart"])
                    ).isoformat(),
                }
            )

    window_starts = [int(row["WindowStart"]) for row in results]
    correlations = [float(row["AverageCorrelation"]) for row in results]
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        [month_start(month) for month in window_starts],
        correlations,
        linewidth=1.3,
    )
    axis.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axis.set(
        xlabel="Start of rolling window",
        ylabel="Average cross-company correlation",
        title=(
            "One-month-lag monthly-return correlation through time\n"
            f"{window_months}-month rolling windows across {company_count} companies"
        ),
        ylim=(-1, 1),
    )
    axis.grid(True, alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(
        output_dir / "rolling-one-month-lag-correlations.png",
        dpi=180,
    )
    plt.close(figure)


def write_averaged_day_results(
    output_dir: Path,
    window_start: int,
    window_years: int,
    results: list[dict[str, float | int]],
    company_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    window_end = window_start + window_years * 12
    csv_path = output_dir / "average-correlation-by-first-x-trading-days.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "AveragedFirstTradingDays",
            "AverageCorrelation",
            "CorrelatedCompanyPairs",
            "MeanMonthlyObservationsPerPair",
            "WindowStart",
            "WindowEndExclusive",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    **row,
                    "WindowStart": month_start(window_start).isoformat(),
                    "WindowEndExclusive": month_start(window_end).isoformat(),
                }
            )

    averaged_days = [int(row["AveragedFirstTradingDays"]) for row in results]
    average_correlations = [
        float(row["AverageCorrelation"]) for row in results
    ]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(averaged_days, average_correlations, marker="o", linewidth=1.8)
    axis.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axis.set(
        xlabel="First trading days per month averaged",
        ylabel="Average cross-company correlation",
        title=(
            "Correlation of monthly returns from first-x-day price averages\n"
            f"{company_count} companies, "
            f"{month_start(window_start):%Y-%m} to "
            f"{month_start(window_end - 1):%Y-%m}"
        ),
        xlim=(0.5, len(averaged_days) + 0.5),
        ylim=(-1, 1),
        xticks=averaged_days,
    )
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_dir / "average-correlation-by-first-x-trading-days.png",
        dpi=180,
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot average cross-company correlation of monthly simple returns "
            "at month lags 0 through 24 over all available data, and plot the "
            "one-month lag through time."
        )
    )
    parser.add_argument("--cohort", type=Path, default=COHORT_PATH)
    parser.add_argument("--monthly-data-dir", type=Path, default=MONTHLY_DATA_DIR)
    parser.add_argument("--daily-data-dir", type=Path, default=DAILY_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-years", type=int, default=DEFAULT_WINDOW_YEARS)
    parser.add_argument(
        "--max-lag-months", type=int, default=DEFAULT_MAX_LAG_MONTHS
    )
    parser.add_argument(
        "--max-averaged-days", type=int, default=DEFAULT_MAX_AVERAGED_DAYS
    )
    parser.add_argument(
        "--rolling-window-months",
        type=int,
        default=DEFAULT_ROLLING_WINDOW_MONTHS,
    )
    args = parser.parse_args()
    if args.window_years <= 0:
        parser.error("--window-years must be positive.")
    if args.max_lag_months <= 0:
        parser.error("--max-lag-months must be positive.")
    if args.max_averaged_days <= 0:
        parser.error("--max-averaged-days must be positive.")
    if args.rolling_window_months < 2:
        parser.error("--rolling-window-months must be at least 2.")

    cohort = load_cohort(args.cohort)
    monthly_returns = load_monthly_returns(cohort, args.monthly_data_dir)
    (
        monthly_window_start,
        monthly_window_end,
        monthly_results,
    ) = average_lagged_correlations(
        monthly_returns,
        args.max_lag_months,
    )
    write_results(
        args.output_dir,
        monthly_window_start,
        monthly_window_end,
        monthly_results,
        len(cohort),
    )
    _, _, rolling_results = rolling_one_month_lag_correlations(
        monthly_returns,
        args.rolling_window_months,
    )
    write_rolling_results(
        args.output_dir,
        rolling_results,
        args.rolling_window_months,
        len(cohort),
    )
    daily_prices = load_daily_prices(cohort, args.daily_data_dir)
    daily_results = average_correlations_by_averaged_days(
        daily_prices,
        monthly_window_start,
        args.window_years,
        args.max_averaged_days,
    )
    write_averaged_day_results(
        args.output_dir,
        monthly_window_start,
        args.window_years,
        daily_results,
        len(cohort),
    )
    print(
        f"Analyzed monthly returns for {len(cohort)} companies over "
        f"{month_start(monthly_window_start):%Y-%m} through "
        f"{month_start(monthly_window_end - 1):%Y-%m}. The initial-days "
        f"comparison uses {args.window_years} years. Plots and correlation "
        f"tables saved to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
