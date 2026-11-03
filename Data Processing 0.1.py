"""Create a sector-grouped correlation heat map for a longest-history cohort.

By default, sectors are ranked by the mean observation count of their five
longest-history companies. Five sectors and five companies per sector are
selected. Ties are resolved alphabetically for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import date
import math
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "Results 0.1"
COMPANY_SUMMARY = DATA_DIR / "company-summary-prep.csv"
COMPANY_DATA_DIR = DATA_DIR / "companies-prep"
TRADING_CALENDAR = DATA_DIR / "sp500_daily_50y.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def parse_iso_date(value: str, source: Path, row_number: int) -> date:
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError as exc:
        raise ValueError(
            f"{source}, row {row_number}: invalid date {value!r}"
        ) from exc


def select_cohort(
    metadata_rows: list[dict[str, str]],
    sector_count: int,
    companies_per_sector: int,
) -> tuple[list[dict[str, str]], dict[str, float]]:
    by_sector: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata_rows:
        ticker = row.get("Ticker", "").strip()
        sector = row.get("Sector", "").strip() or "Unknown"
        try:
            observations = int(row["NumberOfObservations"])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Company summary must contain integer NumberOfObservations."
            ) from exc
        if ticker and observations > 0:
            normalized = dict(row)
            normalized["Sector"] = sector
            by_sector[sector].append(normalized)

    sector_candidates: list[tuple[float, str, list[dict[str, str]]]] = []
    for sector, rows in by_sector.items():
        ranked_rows = sorted(
            rows,
            key=lambda row: (
                -int(row["NumberOfObservations"]),
                row["FirstAvailableTradingDay"],
                row["Ticker"],
            ),
        )
        top_companies = ranked_rows[:companies_per_sector]
        if len(top_companies) < companies_per_sector:
            continue
        score = sum(
            int(row["NumberOfObservations"]) for row in top_companies
        ) / companies_per_sector
        sector_candidates.append((score, sector, top_companies))

    sector_candidates.sort(key=lambda item: (-item[0], item[1].casefold()))
    selected = sector_candidates[:sector_count]
    if len(selected) != sector_count:
        raise ValueError(
            f"Only {len(selected)} sectors have at least "
            f"{companies_per_sector} companies; requested {sector_count}."
        )

    cohort: list[dict[str, str]] = []
    scores: dict[str, float] = {}
    for score, sector, rows in selected:
        scores[sector] = score
        cohort.extend(rows)
    return cohort, scores


def load_returns(
    cohort: list[dict[str, str]], company_data_dir: Path
) -> dict[str, dict[date, float]]:
    returns: dict[str, dict[date, float]] = {}
    for company in cohort:
        ticker = company["Ticker"]
        path = company_data_dir / f"{ticker}-prep.csv"
        data: dict[date, float] = {}
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or "Date" not in reader.fieldnames:
                raise ValueError(f"{path} must have a Date column.")
            if "Return" not in reader.fieldnames:
                raise ValueError(
                    f"{path} does not have a Return column. "
                    'Run "Data Preprocessing.py" first.'
                )
            for row_number, row in enumerate(reader, start=2):
                trading_day = parse_iso_date(row["Date"], path, row_number)
                try:
                    value = float(row["Return"])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    data[trading_day] = value
        if not data:
            raise ValueError(f"No valid returns found for {ticker} in {path}.")
        returns[ticker] = data
    return returns


def find_first_full_common_year(
    returns: dict[str, dict[date, float]], calendar_path: Path
) -> tuple[int, list[date]]:
    calendar_rows = read_csv(calendar_path)
    if not calendar_rows or "Date" not in calendar_rows[0]:
        raise ValueError(f"{calendar_path} must contain a Date column.")

    expected_by_year: dict[int, set[date]] = defaultdict(set)
    for row_number, row in enumerate(calendar_rows, start=2):
        trading_day = parse_iso_date(row["Date"], calendar_path, row_number)
        expected_by_year[trading_day.year].add(trading_day)

    for year in sorted(expected_by_year):
        expected = sorted(expected_by_year[year])
        if all(
            all(trading_day in company_returns for trading_day in expected)
            for company_returns in returns.values()
        ):
            return year, expected

    raise ValueError(
        "No full calendar year was found where all selected companies have "
        "valid daily returns on every S&P 500 trading session."
    )


def write_cohort(
    path: Path,
    cohort: list[dict[str, str]],
    sector_scores: dict[str, float],
) -> None:
    columns = [
        "Sector",
        "SectorMeanObservationsOfTopCompanies",
        "Ticker",
        "Company",
        "FirstAvailableTradingDay",
        "NumberOfObservations",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in cohort:
            writer.writerow(
                {
                    "Sector": row["Sector"],
                    "SectorMeanObservationsOfTopCompanies": (
                        f"{sector_scores[row['Sector']]:.2f}"
                    ),
                    "Ticker": row["Ticker"],
                    "Company": row["Company"],
                    "FirstAvailableTradingDay": row[
                        "FirstAvailableTradingDay"
                    ],
                    "NumberOfObservations": row["NumberOfObservations"],
                }
            )


def plot_heatmap(
    correlation: np.ndarray,
    cohort: list[dict[str, str]],
    year: int,
    observation_count: int,
    output_path: Path,
) -> None:
    ordered_sectors = list(dict.fromkeys(row["Sector"] for row in cohort))
    tickers = [row["Ticker"] for row in cohort]
    boundaries: list[tuple[int, int, str]] = []
    start = 0
    for sector in ordered_sectors:
        size = sum(row["Sector"] == sector for row in cohort)
        boundaries.append((start, start + size, sector))
        start += size

    figure, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(
        correlation,
        cmap="RdBu_r",
        norm=Normalize(vmin=-1, vmax=1),
        interpolation="nearest",
        aspect="equal",
    )
    positions = np.arange(len(tickers))
    axis.set_xticks(positions, labels=tickers, rotation=90, fontsize=8)
    axis.set_yticks(positions, labels=tickers, fontsize=8)
    axis.xaxis.tick_top()
    axis.tick_params(length=0)

    for begin, end, sector in boundaries:
        center = (begin + end - 1) / 2
        axis.text(
            center,
            -1.35,
            textwrap.fill(sector, width=16),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            clip_on=False,
        )
        if begin:
            axis.axvline(begin - 0.5, color="black", linewidth=1.2)
            axis.axhline(begin - 0.5, color="black", linewidth=1.2)

    axis.set_ylim(len(tickers) - 0.5, -2.2)
    axis.set_xlim(-0.5, len(tickers) - 0.5)
    axis.set_title(
        f"Daily simple-return correlation by sector — {year}\n"
        f"First full shared calendar year; {observation_count} common "
        "trading sessions",
        pad=24,
        fontsize=14,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.04)
    colorbar.set_label("Pearson correlation")
    figure.subplots_adjust(left=0.16, right=0.91, bottom=0.10, top=0.82)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sector-grouped return-correlation heat map for the "
            "longest-history companies."
        )
    )
    parser.add_argument("--sectors", type=int, default=5)
    parser.add_argument("--companies-per-sector", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    if args.sectors < 1 or args.companies_per_sector < 2:
        parser.error("--sectors must be positive and --companies-per-sector >= 2.")

    metadata_rows = read_csv(COMPANY_SUMMARY)
    cohort, sector_scores = select_cohort(
        metadata_rows, args.sectors, args.companies_per_sector
    )
    returns = load_returns(cohort, COMPANY_DATA_DIR)
    year, shared_dates = find_first_full_common_year(
        returns, TRADING_CALENDAR
    )
    tickers = [row["Ticker"] for row in cohort]
    matrix = np.array(
        [
            [returns[ticker][trading_day] for ticker in tickers]
            for trading_day in shared_dates
        ],
        dtype=float,
    )
    correlation = np.corrcoef(matrix, rowvar=False)
    if not np.isfinite(correlation).all():
        raise ValueError(
            "Correlation matrix contains undefined values; check for "
            "constant-return company series in the selected year."
        )

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_dir / (
        f"sector-grouped-correlation-heatmap-{year}.png"
    )
    correlation_path = output_dir / (
        f"sector-grouped-correlation-matrix-{year}.csv"
    )
    cohort_path = output_dir / "sector-grouped-analysis-cohort.csv"

    with correlation_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Ticker", *tickers])
        writer.writerows(
            [ticker, *correlation[index].tolist()]
            for index, ticker in enumerate(tickers)
        )
    write_cohort(cohort_path, cohort, sector_scores)
    plot_heatmap(
        correlation, cohort, year, len(shared_dates), heatmap_path
    )

    print(f"Selected {len(cohort)} companies across {args.sectors} sectors.")
    print(
        "Selected sectors: "
        + ", ".join(dict.fromkeys(row["Sector"] for row in cohort))
    )
    print(
        f"First full shared year: {year} "
        f"({len(shared_dates)} common trading sessions)."
    )
    print(f"Heat map: {heatmap_path}")
    print(f"Correlation matrix: {correlation_path}")
    print(f"Cohort details: {cohort_path}")


if __name__ == "__main__":
    main()
