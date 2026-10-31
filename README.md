# Dynamic Financial Networks of the S&P 500

Joint project for **Complex Networks / Complex Systems** and **Pattern Recognition** at the University of Bologna.

**Author:** Leonard Hell
**Status:** Project definition; implementation and empirical results pending.

## Overview

This project investigates how statistical dependencies between stock returns can be represented as a time-evolving network. Each node represents a company, and each edge represents an estimated statistical relationship between two companies' returns.

Networks will be inferred from historical financial time series using sliding time windows. Comparing these networks will make it possible to study changes in market structure, the roles of individual companies, and the emergence of groups of companies with similar behavior.

The project combines statistical learning of dependency structures with graph-theoretical analysis. An additional exploratory objective is to test whether network structure contains information about future market behavior.

## Scope

The intended target is the full S&P 500, subject to historical data availability and computational feasibility.

The analysis pipeline should support different company universes without requiring a redesign. A smaller subset may be used during development before scaling to the full dataset.

## Research questions

1. **Network construction:** How do networks inferred using correlation and mutual information differ?
2. **Centrality:** Which companies occupy central positions, and how do their rankings depend on the centrality measure and change over time?
3. **Community structure:** Do inferred communities correspond to industrial sectors, and how stable are these communities across time windows?
4. **Sector relationships:** Do companies within the same sector predominantly move together, or do some exhibit opposing return patterns?
5. **Market phases:** Do clustering, connectivity, centrality, or robustness display characteristic changes during different market phases?
6. **Predictive information:** Do network-derived features improve predictions of a defined future market outcome relative to suitable baselines?

## Planned methodology

### 1. Data collection and preprocessing

- Obtain historical stock prices for the selected company universe and corresponding industrial-sector labels.
- Align trading dates, document missing-data handling, and use prices adjusted for corporate actions where available.
- Convert prices into returns before estimating dependencies.
- Specify the observation period, sampling frequency, and company-selection criteria.
- Document whether the dataset uses a fixed constituent list or historical index membership, including the resulting survivorship-bias limitations.

The historical constituent data provider, sampling frequency, and exact constituent-selection policy remain to be selected.

#### Index and constituent-data downloader

[`Data Collection.py`](./Data%20Collection.py) downloads daily history for Yahoo Finance's `^GSPC` S&P 500 index symbol using `yfinance`. It requests the 50 calendar years preceding the run date, writes the returned OHLCV data to `data/sp500_daily_50y.csv`, and accepts `--output` to choose another CSV path. The provider may not have data for every requested date; the script reports the actual first and last dates returned.

Install the dependency from the project directory:

```powershell
python -m pip install -r requirements.txt
```

Running `python "Data Collection.py"` downloads the current S&P 500 constituents, including their current GICS sectors, then writes one CSV per company. The script fetches the current list from Wikipedia's S&P 500 constituents page with a normal identified request. It also merges historical membership data from `data\sp500_historical_constituents.csv` if that optional file exists. To download only the index series, run `python "Data Collection.py" --index`.

To include former and delisted companies and accurately identify historical index membership, provide a historical membership manifest with one row per S&P 500 membership interval. The manifest is optional for downloading current members, but required to include the historical universe. It must have these columns:

| Column | Meaning |
| --- | --- |
| `symbol` | Yahoo Finance ticker symbol |
| `company` | Company name |
| `delisted` | `true` or `false`; this is separate from leaving the index |
| `sector` | Company's main sector; optional, defaults to `Unknown` |
| `membership_start` | First membership date in `YYYY-MM-DD` format |
| `membership_end` | Last membership date in `YYYY-MM-DD` format; leave blank for a current membership interval |

If a company entered the index more than once, include a row for each interval using the same ticker and consistent company/delisted/sector values. Use a single consistent classification system (for example, GICS) across the manifest. Save it as `data\sp500_historical_constituents.csv` for automatic merging, or pass its path explicitly:

```powershell
python "Data Collection.py" --constituents path\to\your_constituents.csv
```

The script downloads each distinct symbol once to `data/companies/<symbol>.csv`. Every file includes `Ticker`, `Company`, `Sector`, `Delisted`, `MembershipSource`, and `InIndex`. For tickers in a historical manifest, `InIndex` marks the supplied membership intervals. For current-list-only tickers, earlier `InIndex` values are blank (unknown) rather than falsely labeling each company as a constituent for the full 50 years. Sector is a static label, not a historically time-varying classification; use a dated sector source if the analysis needs historical sector changes. Missing sector values are written as `Unknown`. The company output directory can be changed with `--company-output`.

Without a historical manifest, former/delisted companies are not included. The current constituent list and the `^GSPC` index history do not provide a complete 50-year membership record. To cover the historical universe without survivorship bias, populate the manifest from a reliable point-in-time constituent source that includes removed companies and delisted securities. Yahoo Finance may not have complete history for every delisted ticker; symbols with no returned history are reported, and the script exits unsuccessfully rather than implying the download is complete.

### 2. Dependency estimation and network construction

- Estimate pairwise dependencies within each historical time window.
- Compare correlation-based and mutual-information-based approaches.
- Define a consistent rule for translating dependency estimates into graph edges and weights.
- Specify the treatment of negative correlations and how graph density is controlled when comparing methods.
- Repeat the procedure across sliding windows to obtain a sequence of networks.

The correlation estimator, mutual-information estimator, window length, step size, and graph-filtering rule remain methodological choices to be evaluated and documented.

### 3. Network analysis

Analyze and compare the inferred networks using:

- **Centrality measures**, including **Laplacian Spectral Centrality**, alongside suitable standard centrality measures.
- **Clustering and connectivity** to characterize local and global organization.
- **Community detection** and comparison with industrial-sector labels.
- **Network robustness**, using a documented node- or edge-removal procedure.
- **Temporal comparisons** of network measures, centrality rankings, and community structure.


### 4. Exploratory prediction

If the core analysis is complete, test whether network-derived features contain predictive information about a clearly specified future market outcome, such as future volatility or a market-regime label.

- Define the prediction target and forecast horizon before evaluating models.
- Construct features using only information available at the prediction time.
- Use chronological training, validation, and test periods, accounting for overlap between windows and forecast horizons.
- Compare against simple baselines and, where appropriate, models using return-based features without network information.

The purpose is to assess incremental predictive information. Successful forecasting is not assumed.

## Connection to the two courses

| Course | Project contribution |
| --- | --- |
| Pattern Recognition | Statistical estimation of dependencies from multivariate time series; comparison of network representations; identification of structural patterns; exploratory prediction using network-derived features. |
| Complex Networks / Complex Systems | Centrality, including Laplacian Spectral Centrality; clustering; communities; robustness; temporal evolution of financial networks. |

## Core work and extensions

The **core scope** is a reproducible pipeline for constructing dynamic financial networks, comparing correlation and mutual information, and analyzing centrality, communities, sector relationships, clustering, and robustness over time.

The **predictive analysis is an exploratory extension**, to be pursued after the core pipeline and descriptive comparisons are established. Broader datasets and additional dependency estimators are possible later extensions.

Return co-movement and negative dependence will be interpreted as statistical patterns. They do not, by themselves, establish economic cooperation, competition, or causal influence between companies. Mutual information measures dependence but does not directly distinguish positive from negative relationships.

## Expected deliverables

- A documented, reproducible data-preprocessing and network-construction pipeline.
- A comparison of the selected dependency measures and resulting networks.
- Figures and summaries of centrality, communities, sector patterns, robustness, and temporal evolution.
- A discussion of sensitivity to key methodological choices and limitations of the dataset and inferred relationships.
- If completed, an out-of-sample evaluation of the predictive extension.
- A final report or presentation linking the findings to both courses, in the format agreed with the instructors.

## Current status

This README describes the intended research scope based on the project-proposal correspondence. It does not report completed analyses or results. Dataset details, implementation choices, execution instructions, and references will be added as the project develops.
