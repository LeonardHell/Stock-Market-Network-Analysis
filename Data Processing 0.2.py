"""Analyze sign-based stock co-movement in rolling one-year windows."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import itertools
import math
from pathlib import Path
import shutil
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import networkx as nx
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
COHORT_PATH = DATA_DIR / "Results 0.1" / "sector-grouped-analysis-cohort.csv"
COMPANY_DATA_DIR = DATA_DIR / "companies-prep"
TRADING_CALENDAR = DATA_DIR / "sp500_daily_50y.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "Results 0.2" / "Sign-Based Co-Movement"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def parse_date(value: str, source: Path, row_number: int) -> date:
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError as exc:
        raise ValueError(
            f"{source}, row {row_number}: invalid date {value!r}"
        ) from exc


def load_cohort(path: Path) -> list[dict[str, str]]:
    cohort = read_csv(path)
    if not cohort:
        raise ValueError(f"No companies found in cohort file: {path}")
    required = {"Sector", "Ticker", "Company"}
    if not required.issubset(cohort[0]):
        raise ValueError(
            f"{path} must contain columns {', '.join(sorted(required))}."
        )

    ordered = sorted(
        cohort,
        key=lambda row: (
            row["Sector"].casefold(),
            row["Ticker"].casefold(),
        ),
    )
    tickers = [row["Ticker"].strip() for row in ordered]
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"{path} contains duplicate tickers.")
    return ordered


def load_signs(
    cohort: list[dict[str, str]], source_dir: Path
) -> dict[str, dict[date, int]]:
    signs: dict[str, dict[date, int]] = {}
    for company in cohort:
        ticker = company["Ticker"].strip()
        path = source_dir / f"{ticker}-prep.csv"
        data: dict[date, int] = {}
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
                trading_day = parse_date(row["Date"], path, row_number)
                try:
                    value = float(row["Return"])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    data[trading_day] = (value > 0) - (value < 0)
        if not data:
            raise ValueError(f"No valid returns found for {ticker} in {path}.")
        signs[ticker] = data
    return signs


def load_calendar(path: Path) -> list[date]:
    rows = read_csv(path)
    if not rows or "Date" not in rows[0]:
        raise ValueError(f"{path} must contain a Date column.")
    dates = sorted({
        parse_date(row["Date"], path, row_number)
        for row_number, row in enumerate(rows, start=2)
    })
    if not dates:
        raise ValueError(f"No trading dates found in {path}.")
    return dates


def shift_month(year: int, month: int, offset: int) -> tuple[int, int]:
    month_index = year * 12 + month - 1 + offset
    shifted_year, zero_based_month = divmod(month_index, 12)
    return shifted_year, zero_based_month + 1


def shift_year(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def rolling_windows(
    calendar: list[date],
    signs: dict[str, dict[date, int]],
    window_months: int,
) -> list[tuple[date, date, list[date]]]:
    sessions_by_month: dict[tuple[int, int], list[date]] = {}
    for trading_day in calendar:
        sessions_by_month.setdefault(
            (trading_day.year, trading_day.month), []
        ).append(trading_day)

    month_keys = sorted(sessions_by_month)
    windows: list[tuple[date, date, list[date]]] = []
    have_full_initial_window = False
    all_tickers = list(signs)

    for end_month in month_keys:
        start_month = shift_month(
            end_month[0], end_month[1], -(window_months - 1)
        )
        if start_month not in sessions_by_month:
            continue
        start_date = sessions_by_month[start_month][0]
        end_date = sessions_by_month[end_month][-1]
        expected = [
            trading_day
            for trading_day in calendar
            if start_date <= trading_day <= end_date
        ]
        common_dates = [
            trading_day
            for trading_day in expected
            if all(trading_day in signs[ticker] for ticker in all_tickers)
        ]

        if not have_full_initial_window:
            if len(common_dates) != len(expected):
                continue
            have_full_initial_window = True
        if len(common_dates) < 2:
            continue
        windows.append((start_date, end_date, common_dates))

    if not windows:
        raise ValueError(
            f"No complete initial {window_months}-month window was found "
            "for all companies in the cohort."
        )
    return windows


def calculate_correlation(
    cohort: list[dict[str, str]],
    signs: dict[str, dict[date, int]],
    trading_dates: list[date],
) -> np.ndarray:
    tickers = [company["Ticker"].strip() for company in cohort]
    observations = np.asarray(
        [
            [signs[ticker][trading_day] for ticker in tickers]
            for trading_day in trading_dates
        ],
        dtype=float,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.corrcoef(observations, rowvar=False)


def pair_records(
    cohort: list[dict[str, str]], correlation: np.ndarray
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for left_index, right_index in itertools.combinations(
        range(len(cohort)), 2
    ):
        value = float(correlation[left_index, right_index])
        left = cohort[left_index]
        right = cohort[right_index]
        records.append({
            "left_index": left_index,
            "right_index": right_index,
            "ticker_1": left["Ticker"].strip(),
            "company_1": left["Company"].strip(),
            "sector_1": left["Sector"].strip(),
            "ticker_2": right["Ticker"].strip(),
            "company_2": right["Company"].strip(),
            "sector_2": right["Sector"].strip(),
            "correlation": value,
        })
    return records


def plot_window_matrix(
    correlation: np.ndarray,
    cohort: list[dict[str, str]],
    window: tuple[date, date, list[date]],
    output_path: Path,
) -> None:
    tickers = [row["Ticker"].strip() for row in cohort]
    sectors = [row["Sector"].strip() for row in cohort]
    fig, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(
        correlation,
        cmap="RdBu_r",
        norm=Normalize(vmin=-1, vmax=1),
        interpolation="nearest",
    )
    positions = np.arange(len(tickers))
    axis.set_xticks(positions, labels=tickers, rotation=90, fontsize=8)
    axis.set_yticks(positions, labels=tickers, fontsize=8)
    axis.xaxis.tick_top()
    axis.tick_params(length=0)

    boundaries: list[tuple[int, int, str]] = []
    start = 0
    for sector in dict.fromkeys(sectors):
        end = start + sectors[start:].count(sector)
        boundaries.append((start, end, sector))
        start = end
    for begin, end, sector in boundaries:
        center = (begin + end - 1) / 2
        axis.text(
            center,
            -1.3,
            textwrap.fill(sector, width=16),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            clip_on=False,
        )
        if begin:
            axis.axvline(begin - 0.5, color="black", linewidth=1.1)
            axis.axhline(begin - 0.5, color="black", linewidth=1.1)
    axis.set_ylim(len(tickers) - 0.5, -2.0)
    axis.set_xlim(-0.5, len(tickers) - 0.5)
    axis.set_title(
        "Return-sign correlation, trailing window\n"
        f"{window[0]} to {window[1]} ({len(window[2])} common sessions)",
        pad=22,
    )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.04)
    colorbar.set_label("Pearson correlation of return signs")
    fig.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=0.82)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def select_top_edges(
    pair_rows: list[dict[str, object]], fraction: float
) -> list[dict[str, object]]:
    valid_pairs = [
        row
        for row in pair_rows
        if math.isfinite(float(row["correlation"]))
    ]
    edge_count = max(1, math.ceil(len(valid_pairs) * fraction))
    return sorted(
        valid_pairs,
        key=lambda row: (
            -float(row["correlation"]),
            str(row["ticker_1"]),
            str(row["ticker_2"]),
        ),
    )[:edge_count]


def plot_network(
    cohort: list[dict[str, str]],
    edges: list[dict[str, object]],
    window: tuple[date, date, list[date]],
    output_path: Path,
) -> None:
    graph = nx.Graph()
    for company in cohort:
        graph.add_node(
            company["Ticker"].strip(),
            company=company["Company"].strip(),
            sector=company["Sector"].strip(),
        )
    for edge in edges:
        graph.add_edge(
            str(edge["ticker_1"]),
            str(edge["ticker_2"]),
            correlation=float(edge["correlation"]),
        )

    sectors = list(dict.fromkeys(row["Sector"].strip() for row in cohort))
    palette = plt.get_cmap("tab10")
    sector_colors = {
        sector: palette(index % 10) for index, sector in enumerate(sectors)
    }
    node_colors = [
        sector_colors[graph.nodes[node]["sector"]]
        for node in graph.nodes
    ]
    position = nx.circular_layout(graph)

    positive_edges = [
        (left, right)
        for left, right, data in graph.edges(data=True)
        if data["correlation"] >= 0
    ]
    negative_edges = [
        (left, right)
        for left, right, data in graph.edges(data=True)
        if data["correlation"] < 0
    ]
    edge_widths = {
        (left, right): 0.8 + 5.0 * abs(data["correlation"])
        for left, right, data in graph.edges(data=True)
    }

    figure, axis = plt.subplots(figsize=(14, 11))
    nx.draw_networkx_edges(
        graph,
        position,
        edgelist=positive_edges,
        width=[edge_widths[edge] for edge in positive_edges],
        edge_color="#555555",
        alpha=0.58,
        ax=axis,
    )
    nx.draw_networkx_edges(
        graph,
        position,
        edgelist=negative_edges,
        width=[edge_widths[edge] for edge in negative_edges],
        edge_color="#2166ac",
        style="dashed",
        alpha=0.75,
        ax=axis,
    )
    nx.draw_networkx_nodes(
        graph,
        position,
        node_color=node_colors,
        node_size=700,
        edgecolors="white",
        linewidths=1.1,
        ax=axis,
    )
    nx.draw_networkx_labels(
        graph,
        position,
        font_size=7,
        font_weight="bold",
        ax=axis,
    )

    sector_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=sector_colors[sector],
            markeredgecolor="white",
            markersize=10,
            label=sector,
        )
        for sector in sectors
    ]
    edge_handles = [
        Line2D([0], [0], color="#555555", linewidth=2, label="Positive"),
        Line2D(
            [0],
            [0],
            color="#2166ac",
            linewidth=2,
            linestyle="--",
            label="Negative",
        ),
    ]
    sector_legend = axis.legend(
        handles=sector_handles,
        title="Company sector",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        borderaxespad=0,
    )
    axis.add_artist(sector_legend)
    axis.legend(
        handles=edge_handles,
        title="Selected edge sign",
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        borderaxespad=0,
    )
    axis.set_title(
        "Selected stock-pair sign correlations\n"
        f"Trailing window {window[0]} to {window[1]}; "
        f"{len(edges)} of {len(cohort) * (len(cohort) - 1) // 2} pairs",
        pad=18,
    )
    axis.axis("off")
    figure.text(
        0.76,
        0.035,
        "Edge width scales with |correlation|; dashed blue edges are negative.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.03, right=0.68, top=0.9, bottom=0.08)
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def render_window_frames(
    cohort: list[dict[str, str]],
    windows: list[tuple[date, date, list[date]]],
    correlations: list[np.ndarray],
    pair_rows_by_window: list[list[dict[str, object]]],
    edge_fraction: float,
    heatmap_dir: Path,
    network_dir: Path,
) -> tuple[Path, Path]:
    if not (
        len(windows) == len(correlations) == len(pair_rows_by_window)
    ):
        raise ValueError(
            "Each rolling window must have a correlation matrix and pair rows."
        )
    for directory in (heatmap_dir, network_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for old_frame in directory.glob("window-*.png"):
            old_frame.unlink()

    latest_heatmap: Path | None = None
    latest_network: Path | None = None
    for index, (window, correlation, pair_rows) in enumerate(
        zip(windows, correlations, pair_rows_by_window),
        start=1,
    ):
        frame_name = (
            f"window-{index:05d}_{window[0]}_{window[1]}.png"
        )
        heatmap_path = heatmap_dir / frame_name
        network_path = network_dir / frame_name
        plot_window_matrix(correlation, cohort, window, heatmap_path)
        edges = select_top_edges(pair_rows, edge_fraction)
        plot_network(cohort, edges, window, network_path)
        latest_heatmap = heatmap_path
        latest_network = network_path

    if latest_heatmap is None or latest_network is None:
        raise ValueError("Cannot render rolling visualizations without windows.")
    return latest_heatmap, latest_network


def write_pair_history(
    path: Path,
    pair_rows_by_window: list[list[dict[str, object]]],
    windows: list[tuple[date, date, list[date]]],
) -> None:
    fields = [
        "WindowStart",
        "WindowEnd",
        "CommonTradingSessions",
        "Sector1",
        "Ticker1",
        "Company1",
        "Sector2",
        "Ticker2",
        "Company2",
        "SignCorrelation",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for window, pair_rows in zip(windows, pair_rows_by_window):
            for row in pair_rows:
                correlation = float(row["correlation"])
                writer.writerow({
                    "WindowStart": window[0].isoformat(),
                    "WindowEnd": window[1].isoformat(),
                    "CommonTradingSessions": len(window[2]),
                    "Sector1": row["sector_1"],
                    "Ticker1": row["ticker_1"],
                    "Company1": row["company_1"],
                    "Sector2": row["sector_2"],
                    "Ticker2": row["ticker_2"],
                    "Company2": row["company_2"],
                    "SignCorrelation": (
                        f"{correlation:.12g}"
                        if math.isfinite(correlation)
                        else "NaN"
                    ),
                })


def write_edges(path: Path, edges: list[dict[str, object]]) -> None:
    fields = [
        "Ticker1",
        "Company1",
        "Sector1",
        "Ticker2",
        "Company2",
        "Sector2",
        "SignCorrelation",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in edges:
            writer.writerow({
                "Ticker1": row["ticker_1"],
                "Company1": row["company_1"],
                "Sector1": row["sector_1"],
                "Ticker2": row["ticker_2"],
                "Company2": row["company_2"],
                "Sector2": row["sector_2"],
                "SignCorrelation": f"{float(row['correlation']):.12g}",
            })


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze co-direction of daily stock returns in monthly-rolling "
            "one-year windows."
        )
    )
    parser.add_argument("--window-months", type=int, default=12)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument(
        "--analysis-years",
        type=int,
        default=3,
        help="Limit analysis to this many years from the first shared session.",
    )
    parser.add_argument("--edge-fraction", type=float, default=0.10)
    parser.add_argument("--cohort", type=Path, default=COHORT_PATH)
    parser.add_argument("--company-data", type=Path, default=COMPANY_DATA_DIR)
    parser.add_argument("--calendar", type=Path, default=TRADING_CALENDAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.window_months < 1 or args.step_months < 1 or args.analysis_years < 1:
        parser.error(
            "--window-months, --step-months, and --analysis-years "
            "must be positive."
        )
    if not 0 < args.edge_fraction <= 1:
        parser.error("--edge-fraction must be greater than 0 and at most 1.")

    cohort = load_cohort(args.cohort.expanduser())
    signs = load_signs(cohort, args.company_data.expanduser())
    calendar = load_calendar(args.calendar.expanduser())
    windows = rolling_windows(calendar, signs, args.window_months)
    all_tickers = list(signs)
    mutual_dates = [
        trading_day
        for trading_day in calendar
        if all(trading_day in signs[ticker] for ticker in all_tickers)
    ]
    analysis_end = shift_year(mutual_dates[0], args.analysis_years)
    windows = [window for window in windows if window[1] <= analysis_end]
    if not windows:
        raise ValueError(
            f"No complete {args.window_months}-month windows fit within the "
            f"first {args.analysis_years} years of shared data."
        )
    if args.step_months > 1:
        stepped_windows = windows[::args.step_months]
        if stepped_windows[-1] != windows[-1]:
            stepped_windows.append(windows[-1])
        windows = stepped_windows

    sector_order = list(dict.fromkeys(row["Sector"].strip() for row in cohort))
    pair_rows_by_window: list[list[dict[str, object]]] = []
    correlations: list[np.ndarray] = []
    for _, _, common_dates in windows:
        correlation = calculate_correlation(cohort, signs, common_dates)
        correlations.append(correlation)
        pair_rows_by_window.append(pair_records(cohort, correlation))

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_dir / "rolling-heatmaps"
    network_dir = output_dir / "rolling-networks"
    rolling_heatmap = output_dir / "rolling-sign-correlation-heatmap.png"
    latest_heatmap = output_dir / "latest-sign-correlation-heatmap.png"
    network_path = output_dir / "latest-sign-correlation-network.png"
    history_path = output_dir / "rolling-sign-correlations.csv"
    edges_path = output_dir / "latest-network-edges.csv"

    latest_heatmap_frame, latest_network_frame = render_window_frames(
        cohort,
        windows,
        correlations,
        pair_rows_by_window,
        args.edge_fraction,
        heatmap_dir,
        network_dir,
    )
    latest_window = windows[-1]
    latest_pairs = pair_rows_by_window[-1]
    latest_edges = select_top_edges(latest_pairs, args.edge_fraction)
    shutil.copyfile(latest_heatmap_frame, latest_heatmap)
    shutil.copyfile(latest_heatmap_frame, rolling_heatmap)
    shutil.copyfile(latest_network_frame, network_path)
    write_pair_history(history_path, pair_rows_by_window, windows)
    write_edges(edges_path, latest_edges)

    print(f"Companies: {len(cohort)} across {len(sector_order)} sectors.")
    print(
        f"Rolling windows: {len(windows)}; each spans "
        f"{args.window_months} months and advances {args.step_months} month(s)."
    )
    print(
        f"Window range: {windows[0][0]}–{windows[0][1]} through "
        f"{latest_window[0]}–{latest_window[1]}."
    )
    print(
        f"Analysis limited to the first {args.analysis_years} years of shared "
        f"data, through {analysis_end}."
    )
    print(
        f"Latest window uses {len(latest_window[2])} common trading sessions; "
        f"selected {len(latest_edges)} highest-correlation edges."
    )
    print(f"Rolling heat maps: {heatmap_dir}")
    print(f"Rolling network graphs: {network_dir}")
    print(f"Latest correlation matrix: {latest_heatmap}")
    print(f"Latest network graph: {network_path}")
    print(f"Rolling pair correlations: {history_path}")
    print(f"Latest edge list: {edges_path}")


if __name__ == "__main__":
    main()
