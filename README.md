# ArthaDrishti Portfolio Studio

**ArthaDrishti** (Sanskrit: अर्थ दृष्टि — "wealth vision") is an interactive Python web application that blends the **Black-Litterman model** with real-time market data, investor views, and rich Plotly visualizations to construct and analyze optimized equity portfolios.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-deployed-app-url.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage Guide](#usage-guide)
6. [Core Modules](#core-modules)
   - [Configuration (`config/settings.py`)](#configuration-configsettingspy)
   - [Data Ingestion (`data/yahoo_data.py`)](#data-ingestion-datayahoo_datapy)
   - [Black-Litterman Model (`models/black_litterman.py`)](#black-litterman-model-modelsblack_littermanpy)
   - [Portfolio Optimization (`models/optimization.py`)](#portfolio-optimization-modelsoptimizationpy)
   - [Helper Utilities (`utils/helpers.py`)](#helper-utilities-utilshelperspy)
   - [Visualization Engine (`utils/visualization.py`)](#visualization-engine-utilsvisualizationpy)
   - [Main Application (`app.py`)](#main-application-apppy)
7. [Black-Litterman Model Walkthrough](#black-litterman-model-walkthrough)
8. [Portfolio Optimization Details](#portfolio-optimization-details)
9. [Visualization Gallery](#visualization-gallery)
10. [Key Algorithms](#key-algorithms)
11. [Troubleshooting](#troubleshooting)
12. [Development](#development)
13. [Deployment](#deployment)
14. [License](#license)
15. [References](#references)

---

## Overview

### What Is ArthaDrishti?

ArthaDrishti Portfolio Studio is a comprehensive portfolio management platform that addresses the well-known **estimation error problem** in classical Markowitz mean-variance optimization. By anchoring portfolio construction on **market equilibrium returns** (implied from market capitalizations) and blending in **subjective investor views** via the Black-Litterman Bayesian framework, the application produces more stable, intuitive, and robust optimal portfolios.

### Core Value Proposition

| Problem | Traditional MVO | ArthaDrishti (Black-Litterman) |
|---|---|---|
| Input returns | Sample mean (noisy) | Market-implied + investor views |
| Sensitivity to estimation error | High | Low (anchored to market) |
| Intuitive portfolio weights | No | Yes (starts from market cap) |
| Investor views integration | None | Full Bayesian view support |
| Visualization | Static charts | 70+ interactive Plotly charts |

### Feature Highlights

- **100+ Equity Universe**: Technology, finance, healthcare, energy, industrials, consumer, and communications sectors
- **Real-time Yahoo Finance Integration**: OHLCV data, market capitalization retrieval, automatic ticker validation
- **Black-Litterman Model**: Full implementation with absolute and relative view support, confidence weighting, posterior computation
- **Mean-Variance Optimization**: SLSQP solver with configurable constraints (short-selling, max weight per asset)
- **Efficient Frontier**: 50-point frontier computation with Sharpe ratio and minimum-volatility portfolios
- **70+ Interactive Visualizations**: Portfolio analytics, technical indicators, risk metrics, statistical charts, 3D visualizations
- **Results Export**: CSV download for results summary table, portfolio metrics, and comparison data
- **Professional Dark Theme**: Custom Streamlit CSS with gradient backgrounds, styled tabs, metrics, and sidebar

---

## Architecture

The application follows a **layered modular architecture**:

```
ArthaDrishti Portfolio Studio
├── Presentation Layer          (app.py)
│   Streamlit UI: 4 tabs, sidebar, session state management
│
├── Visualization Layer         (utils/visualization.py)
│   Visualizer class: 70+ Plotly charts with dark theme
│
├── Business Logic Layer        (models/)
│   ├── BlackLittermanModel     (black_litterman.py)
│   │   Prior returns, view management, posterior computation
│   └── PortfolioOptimizer      (optimization.py)
│       Mean-variance optimization, efficient frontier
│
├── Data Access Layer           (data/yahoo_data.py)
│   YahooFinanceData class: data fetching, returns, market caps
│
├── Configuration Layer         (config/settings.py)
│   Defaults, model hyperparameters, theme colors
│
└── Utilities Layer             (utils/helpers.py)
    Formatting, validation, risk statistics
```

### Data Flow

```
User selects tickers → YahooFinanceData.fetch_data() → returns + market_caps
       → BlackLittermanModel(returns, market_caps) → implied returns (Pi)
       → User adds views → calculate_posterior() → posterior returns + covariance
       → PortfolioOptimizer(posterior_returns, posterior_cov) → optimal weights
       → Visualizer → 70+ interactive charts
       → Streamlit UI renders charts + metrics cards
```

---

## Installation

### Prerequisites

- **Python 3.11+** (developed and tested on Python 3.11)
- **Git** for version control
- **pip** package manager

### Setup

```bash
git clone https://github.com/sourishdey2005/ArthaDrishti-Portfolio-Studio.git
cd ArthaDrishti-Portfolio-Studio

# Create and activate virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

The `requirements.txt` file includes:

| Package | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `yfinance` | Yahoo Finance API wrapper |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `plotly` | Interactive data visualization |
| `scipy` | Scientific computing (optimization) |
| `matplotlib` | Static plotting (fallback heatmaps) |
| `seaborn` | Statistical data visualization |
| `PyPortfolioOpt` | Portfolio optimization library |

---

## Configuration

All configuration lives in `config/settings.py`:

### Model Parameters

| Parameter | Default | Description |
|---|---|---|
| `RISK_FREE_RATE` | `0.02` (2%) | Risk-free rate for Sharpe ratio calculations |
| `TAU` | `0.025` | Uncertainty in the prior (scaling factor for Σ in BL formula) |
| `DELTA` | `2.5` | Risk aversion coefficient (links returns to market weights) |

### Optimization Constraints

| Parameter | Default | Description |
|---|---|---|
| `ALLOW_SHORT` | `False` | Short-selling toggle |
| `MAX_WEIGHT` | `0.3` (30%) | Maximum weight per asset |
| `MIN_WEIGHT` | `0.0` | Minimum weight per asset |

### Default Assets

- `DEFAULT_TICKERS`: 10 core tickers (AAPL, GOOGL, MSFT, AMZN, META, JPM, V, JNJ, WMT, PG)
- `TICKER_UNIVERSE_100`: Full 100-ticker universe across all sectors
- `DEFAULT_START_DATE`: `2020-01-01`
- `DEFAULT_END_DATE`: `2023-12-31`

### Visualization Theme

The dark theme uses a professional color palette:

```python
THEME_COLORS = {
    'primary':   '#60a5fa',  # Blue
    'secondary': '#f59e0b',  # Amber
    'success':   '#22c55e',  # Green
    'danger':    '#f87171',  # Red
    'background': '#0f172a', # Dark navy
    'surface':    '#111827', # Slightly lighter
    'grid':      'rgba(148, 163, 184, 0.18)',
    'text':      '#f8fafc',
    'muted_text': '#cbd5e1'
}
```

---

## Usage Guide

### Running the Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### Application Tabs

#### 1. Data Overview Tab

- **Ticker Selection**: Multi-select from 100+ equities or enter custom tickers
- **Date Range**: Select start and end dates (default: 2020-01-01 to 2023-12-31)
- **Model Parameters**: Adjust delta (risk aversion) and tau (prior uncertainty)
- **Market Summary**: Table showing each ticker's mean return, volatility, Sharpe ratio, skewness, and kurtosis

#### 2. Investor Views Tab

- **Absolute Views**: Add a view on a single asset's expected return with confidence level (0-100%)
- **Relative Views**: Add a view on the outperformance between two assets with confidence level
- **View Management**: List all active views, remove individual views, clear all views
- **Posterior Comparison**: Interactive bar chart comparing prior (implied) vs posterior (view-adjusted) returns

#### 3. Results Tab

- **Portfolio Metrics**: Interactive metrics cards (expected return, volatility, Sharpe ratio, diversification ratio)
- **Results Summary Table**: Per-asset table with market weights, prior/posterior returns, optimal weights, volatility
- **Active Weights Chart**: Bar chart comparing optimal portfolio weights vs market-cap-weighted benchmark
- **Return Contribution**: Per-asset contribution to portfolio returns
- **Weight Comparison**: Grouped bar chart (market vs prior vs posterior)
- **Sankey Flow**: Asset allocation flow diagram showing weight transitions

#### 4. Optimization Tab

- **Efficient Frontier**: 3D scatter plot of target returns vs volatility vs Sharpe ratios
- **Risk-Return Lens**: Comprehensive risk analysis including VaR, CVaR, max drawdown, downside deviation
- **Portfolio Risk Decomposition**: Risk contribution by asset, marginal contribution to risk
- **Covariance Surface**: 3D visualization of the covariance matrix structure
- **Weight Heatmap**: Asset weight across different frontier points

### Sidebar Controls

- **Session Management**: Session ID display, reset button
- **Parameter Tuning**: Delta, tau, short-selling toggle, max weight per asset
- **Data Export**: Download results tables as CSV

---

## Core Modules

### Configuration (`config/settings.py`)

The central configuration file defines all application parameters. It is structured into four sections:

1. **Market Settings**: Risk-free rate, tau, delta parameters
2. **Default Assets**: 10 default tickers and 100-ticker universe
3. **Date Range**: Default start/end dates for historical data
4. **Optimization Constraints**: Short-selling, max/min weight bounds
5. **Visualization Theme**: Color palette for all charts

**Key variables:**

```python
RISK_FREE_RATE = 0.02       # Annual risk-free rate
TAU = 0.025                 # Prior uncertainty scaling
DELTA = 2.5                 # Risk aversion coefficient
ALLOW_SHORT = False         # Short-selling disabled by default
MAX_WEIGHT = 0.3            # 30% cap per asset
```

---

### Data Ingestion (`data/yahoo_data.py`)

The `YahooFinanceData` class is the sole data access layer, responsible for:

1. **Ticker Validation**: Strips whitespace, uppercases, deduplicates, and validates against Yahoo Finance
2. **OHLCV Data Fetching**: Uses `yfinance.download()` with `group_by='ticker'` for multi-ticker downloads
3. **Returns Calculation**: Daily percentage returns via `pct_change().dropna()`
4. **Market Capitalization Retrieval**: Fetches via `stock.fast_info` or `stock.info` with fallback to 1B default
5. **Price Normalization**: Normalizes prices to 100 for comparative visualization
6. **OHLC Extraction**: Extracts Open, High, Low, Close, Volume for candlestick and range-based charts

**Key methods:**

| Method | Description |
|---|---|
| `fetch_data()` | Downloads price data, computes returns, retrieves market caps |
| `_get_market_caps()` | Fetches market capitalization for each valid ticker |
| `get_price_history()` | Returns normalized price series (base=100) |
| `get_ohlc_history()` | Returns OHLC dictionary per ticker for technical charts |
| `get_correlation_matrix()` | Computes correlation matrix of returns |
| `get_covariance_matrix()` | Computes annualized covariance matrix (x252) |
| `get_summary_stats()` | Returns mean, std dev, Sharpe, skewness, kurtosis |
| `get_valid_tickers(tickers)` | Static method to validate ticker list |

**Error handling:**

The module handles failures gracefully:
- Invalid tickers are silently dropped with a warning
- Failed market cap lookups default to 1 billion USD
- API errors (timezone issues, JSON decode errors) trigger clear error messages

---

### Black-Litterman Model (`models/black_litterman.py`)

The `BlackLittermanModel` class implements the full Black-Litterman Bayesian framework.

#### Mathematical Foundation

The Black-Litterman model combines:

1. **Prior (Market Implied Returns)**: Computed via reverse optimization:
   ```
   Pi = delta * Sigma * w_mkt
   ```
   Where `Pi` is the vector of implied excess returns, `delta` is the risk aversion coefficient, `Sigma` is the covariance matrix, and `w_mkt` is the vector of market capitalization weights.

2. **Investor Views**: Expressed as linear combinations of asset returns:
   ```
   P * mu + epsilon ~ N(Q, Omega)
   ```
   Where `P` is the pick matrix, `Q` is the view vector, and `Omega` is the view uncertainty matrix.

3. **Posterior**: Computed via Bayesian updating:
   ```
   mu_bl = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]
   Sigma_bl = Sigma + [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
   ```

#### Key Methods

| Method | Description |
|---|---|
| `__init__(returns, market_caps, delta, tau)` | Initializes model, computes covariance and implied returns |
| `_calculate_implied_returns()` | Reverse optimization to compute Pi from market weights |
| `add_absolute_view(asset, expected_return, confidence)` | Adds absolute return view on single asset |
| `add_relative_view(asset_out, asset_under, expected_outperformance, confidence)` | Adds relative outperformance view |
| `calculate_posterior()` | Computes posterior returns and covariance using BL formula |
| `get_optimal_weights(allow_short, max_weight)` | Mean-variance optimization on posterior returns |
| `get_results_dataframe()` | Returns comprehensive results DataFrame |

#### View Types

**Absolute Views**: Specify expected return for a single asset (e.g., "AAPL will return 15% annually")
- P matrix row: `[0, 0, 1, 0, ...]` (1 in the position of the asset)
- Q vector: `[0.15]` (expected return in decimal)

**Relative Views**: Specify outperformance between two assets (e.g., "AAPL will outperform MSFT by 3%")
- P matrix row: `[1, 0, -1, 0, ...]` (1 for outperformer, -1 for underperformer)
- Q vector: `[0.03]` (expected outperformance in decimal)

#### Confidence Mapping

Confidence levels (0-1) are converted to view uncertainty:
```
uncertainty = (1 - confidence) * tau * variance
```
Higher confidence → lower uncertainty → views have greater weight in posterior.

---

### Portfolio Optimization (`models/optimization.py`)

The `PortfolioOptimizer` class implements classical mean-variance optimization using SciPy's SLSQP solver.

#### Optimization Types

| Method | Objective |
|---|---|
| `maximize_sharpe_ratio()` | Maximize (return - risk_free) / volatility |
| `minimize_volatility()` | Minimize portfolio variance subject to weight constraints |
| `optimize_with_target_return()` | Minimize variance for a given target return |
| `get_efficient_frontier()` | Compute frontier across 50 target return levels |

#### Constraints

- **Weight Sum**: `sum(weights) = 1` (fully invested)
- **Short-Selling**: Optional (bounds become `[-max_weight, max_weight]` if allowed)
- **Max Weight**: Per-asset cap (default 30%)
- **Min Weight**: Per-asset floor (default 0%)

#### Portfolio Statistics

`calculate_portfolio_stats()` returns:

| Metric | Formula |
|---|---|
| Return | `sum(weights * expected_returns)` |
| Volatility | `sqrt(w' * Sigma * w)` |
| Sharpe Ratio | `(return - risk_free_rate) / volatility` |
| Diversification Ratio | `sum(w_i * sigma_i) / portfolio_volatility` |

---

### Helper Utilities (`utils/helpers.py`)

Utility functions for formatting and risk metrics:

| Function | Description |
|---|---|
| `format_currency(value, currency)` | Formats numbers as `$1,234.56` or custom currency |
| `format_percentage(value, decimals)` | Formats as `12.34%` with N decimal places |
| `calculate_portfolio_metrics(weights, expected_returns, cov_matrix, risk_free_rate)` | Returns dict with return, volatility, Sharpe, diversification ratio |
| `validate_tickers(tickers)` | Strips, uppercases, deduplicates ticker list |
| `calculate_market_cap_weights(market_caps)` | Converts market caps to normalized weights |
| `calculate_historical_volatility(returns, annualize)` | Annualized standard deviation (x sqrt(252)) |
| `calculate_max_drawdown(prices)` | Maximum drawdown from rolling peaks |
| `calculate_var(returns, confidence_level)` | Value at Risk at 95% confidence |
| `calculate_cvar(returns, confidence_level)` | Conditional Value at Risk (expected shortfall) |
| `calculate_turnover(weights_new, weights_old)` | Portfolio turnover ratio |
| `create_summary_table(results_df)` | Creates summary metrics DataFrame |

---

### Visualization Engine (`utils/visualization.py`)

The `Visualizer` class provides 70+ interactive Plotly charts organized into three categories.

#### Class Structure

```python
class Visualizer:
    def __init__(self, theme_colors=None)
    def _update_layout(self, fig, title, xaxis_title='', yaxis_title='')
    def _portfolio_returns(self, returns_df, weights=None)
    def _drawdown_series(self, returns_series)
    def _summary_stats(self, returns_df, risk_free_rate=0.02)
```

#### 9.1 Core Portfolio Visualizations

| Method | Chart Type | Description |
|---|---|---|
| `plot_weight_comparison()` | Grouped bar chart | Market vs prior vs posterior weights |
| `plot_active_weight_bar()` | Bar chart | Active weights (optimal vs market cap) |
| `plot_sankey_allocation()` | Sankey diagram | Weight flow between market/prior/posterior |
| `plot_cumulative_returns()` | Line chart | Cumulative return comparison |
| `plot_return_contribution()` | Bar chart | Per-asset return contribution |
| `plot_efficient_frontier()` | 3D scatter | Risk-return-volatility frontier |
| `plot_covariance_surface()` | 3D surface | Covariance matrix heatmap |
| `plot_weight_heatmap()` | Heatmap | Asset weights across frontier points |

#### 9.2 Financial Return & Risk Visualizations

| Method | Chart Type | Description |
|---|---|---|
| `plot_calendar_returns()` | Bar chart | Monthly returns heatmap |
| `plot_annual_returns()` | Bar chart | Yearly aggregated returns |
| `plot_quarterly_returns()` | Bar chart | Quarterly return bars |
| `plot_drawdown_chart()` | Area chart | Portfolio drawdown over time |
| `plot_rolling_beta()` | Line chart | Rolling market beta |
| `plot_heatmap()` | Heatmap | Correlation/covariance matrix |
| `plot_scatter_matrix()` | Scatter matrix | Pairwise return relationships |
| `plot_volatility_bubbles()` | Bubble chart | Risk-return-volatility trade-off |
| `plot_tail_risk_chart()` | Violin + box | Distribution tails (VaR/CVaR) |
| `plot_regime_classification()` | Scatter | Return distribution regimes |
| `plot_risk_parity_weights()` | Bar chart | Risk-parity allocation |
| `plot_risk_contribution()` | Bar chart | Per-asset risk contribution |

#### 9.3 Technical Indicator Visualizations (28+ Indicators)

| Category | Indicators | Chart Patterns |
|---|---|---|
| **Momentum** | RSI, Stochastic RSI, CCI, Williams %R, ROC, Momentum, TSI | Oscillators + threshold bands |
| **Trend** | MACD (histogram + signal), ADX, Parabolic SAR, DMI, Aroon | Trend lines + directional indicators |
| **Volatility** | Bollinger Bands, Keltner Channels, Donchian Channels, ATR, Chaikin Volatility | Envelope/band overlays |
| **Volume** | OBV, Volume Rate of Change, Money Flow Index, Chaikin Money Flow | Volume histograms + divergence |
| **Moving Averages** | SMA, EMA, WMA, Hull MA, Triple EMA, VWMA, VWAP | Multi-MA overlays |
| **Advanced** | Ichimoku Cloud (full), Fibonacci Retracement, ZigZag, Pivot Points | Multi-panel complex overlays |

Each technical indicator chart includes:
- Price candles (OHLC) or line
- Indicator subplots
- Threshold reference lines (overbought/oversold)
- Interactive hover tooltips with values
- Custom styling matching the dark theme

#### Theme & Styling

All charts use `_update_layout()` which applies:
- Dark theme template (`plotly_dark`)
- Custom background color (`#0f172a`)
- Plot surface color (`#111827`)
- Grid lines (`rgba(148, 163, 184, 0.18)`)
- Consistent font colors for readability
- Hover mode 'closest' for precise data inspection

---

### Main Application (`app.py`)

The Streamlit application orchestrates the entire workflow:

1. **Session State Management**: Persistent state across reruns using `st.session_state`
2. **Sidebar Controls**: Ticker selection, date range, model parameters, view management
3. **Four Main Tabs**: Data Overview, Investor Views, Results, Optimization
4. **Error Handling**: Graceful error messages for invalid tickers or API failures
5. **Result Export**: CSV download buttons for all data tables

#### Key Data Flow Functions

| Function | Purpose |
|---|---|
| `initialize_session_state()` | Sets default session state values |
| `render_chart_pairs(chart_func, ...)` | Renders two charts side by side |
| `build_analysis_bundle(...)` | Assembles returns, market caps, BL model, optimizer |
| `main()` | Orchestrates the entire application flow |

#### Session State Keys

| Key | Type | Description |
|---|---|---|
| `returns` | pd.DataFrame | Daily returns matrix |
| `market_caps` | pd.Series | Market capitalization per ticker |
| `bl_model` | BlackLittermanModel | Initialized BL model instance |
| `optimizer` | PortfolioOptimizer | Initialized optimizer instance |
| `optimal_weights` | np.array | Optimal portfolio weights |
| `results_df` | pd.DataFrame | Comprehensive results table |

---

## Black-Litterman Model Walkthrough

### Step 1: Market Data Collection

```python
data_fetcher = YahooFinanceData(tickers, start_date, end_date)
returns, market_caps = data_fetcher.fetch_data()
```

The `returns` DataFrame contains daily percentage returns for each ticker. The `market_caps` Series contains each ticker's market capitalization.

### Step 2: Model Initialization

```python
bl_model = BlackLittermanModel(
    returns=returns,
    market_caps=market_caps,
    delta=2.5,    # Risk aversion
    tau=0.025     # Prior uncertainty
)
```

The model computes:
- **Covariance matrix**: From historical returns
- **Market weights**: Normalized from market caps
- **Implied returns (Pi)**: `delta * Σ * w_mkt`

### Step 3: Adding Investor Views

```python
# Absolute view: AAPL expected return = 15%
bl_model.add_absolute_view(
    asset="AAPL",
    expected_return=0.15,
    confidence=0.6
)

# Relative view: AAPL outperforms MSFT by 3%
bl_model.add_relative_view(
    asset_outperform="AAPL",
    asset_underperform="MSFT",
    expected_outperformance=0.03,
    confidence=0.5
)
```

### Step 4: Posterior Computation

```python
posterior_returns, posterior_cov = bl_model.calculate_posterior()
```

The posterior returns blend the market-implied returns with investor views through the Bayesian formula. When no views are added, the posterior equals the prior.

### Step 5: Optimization

```python
optimal_weights = bl_model.get_optimal_weights(
    allow_short=False,
    max_weight=0.3
)
```

The optimal weights are computed via mean-variance optimization on the posterior returns and covariance.

### Step 6: Results Generation

```python
results_df = bl_model.get_results_dataframe()
# Contains: Asset, Market Weight, Prior Return, Posterior Return,
#           Prior Volatility, Posterior Volatility, Optimal Weight
```

---

## Portfolio Optimization Details

### Mean-Variance Framework

The optimization solves:

```
minimize:   w' * Sigma * w      (portfolio variance)
subject to: sum(w) = 1          (fully invested)
            w >= 0              (no shorting, optional)
            w <= max_weight     (per-asset cap)
```

Or for Sharpe maximization:
```
maximize:   (w' * mu - rf) / sqrt(w' * Sigma * w)
```

### Efficient Frontier

The `get_efficient_frontier()` method computes 50 points along the frontier:

1. Determine return range: `[min(return_i), max(return_i)]`
2. For each target return, solve constrained minimization
3. Return arrays: `target_returns`, `volatilities`, `weights`

### Constraint Handling

| Constraint | Type | Implementation |
|---|---|---|
| Weight sum = 1 | Equality | `{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}` |
| No shorting | Inequality | `{'type': 'ineq', 'fun': lambda x: x}` |
| Max weight | Bounds | `[(0, 0.3) for _ in range(n_assets)]` |
| Min weight | Bounds | `[(0.0, 0.3) for _ in range(n_assets)]` |

When `allow_short=True`, bounds become `(-max_weight, max_weight)` and the short-selling constraint is removed.

---

## Visualization Gallery

### Portfolio Analytics

1. **Weight Comparison**: Grouped bar chart comparing market weights, prior-implied weights, and posterior-adjusted weights across all assets
2. **Active Weights**: Bar chart showing the difference between optimal portfolio weights and market-cap-weighted benchmark (green = overweight, red = underweight)
3. **Sankey Flow**: Interactive flow diagram showing how capital allocations shift from market → prior → posterior → optimal
4. **Cumulative Returns**: Time-series of cumulative returns for buy-and-hold vs optimized portfolio
5. **Return Contribution**: Bar chart showing each asset's contribution to total portfolio return
6. **Efficient Frontier (3D)**: Scatter plot of target returns vs volatility vs Sharpe ratio
7. **Covariance Surface**: 3D surface plot of the covariance matrix structure
8. **Weight Heatmap**: Heatmap of asset weights across different frontier points

### Risk & Return Analysis

1. **Calendar Returns**: Monthly returns heatmap with color intensity by return magnitude
2. **Annual Returns**: Year-over-year aggregated return bars
3. **Quarterly Returns**: Quarterly return bars with trend indicators
4. **Drawdown Chart**: Area chart showing portfolio drawdown from peak over time
5. **Rolling Beta**: 60-day rolling beta against market proxy
6. **Correlation Heatmap**: Asset-to-asset correlation matrix
7. **Covariance Heatmap**: Asset-to-asset covariance matrix
8. **Scatter Matrix**: Pairwise scatter plots of asset returns with histograms
9. **Volatility Bubbles**: 3D bubble chart (risk vs return vs diversification)
10. **Tail Risk Chart**: Violin + box plot showing return distribution tails with VaR/CVaR markers
11. **Regime Classification**: Scatter plot classifying return days into bull/bear/volatile regimes
12. **Risk Parity Weights**: Bar chart showing risk-parity optimal allocation
13. **Risk Contribution**: Bar chart showing each asset's contribution to portfolio risk

### Technical Indicators (28+ Indicators)

#### Momentum Indicators
- **RSI**: Relative Strength Index with overbought (70) / oversold (30) thresholds
- **StochRSI**: Stochastic of RSI for momentum confirmation
- **CCI**: Commodity Channel Index for cyclical trends
- **Williams %R**: Overbought/oversold momentum oscillator
- **ROC**: Rate of Change momentum indicator
- **Momentum**: Price momentum over N-period comparison
- **TSI**: True Strength Index for smoothed momentum
- **Awesome Oscillator**: Histogram of momentum acceleration

#### Trend Indicators
- **MACD**: Moving Average Convergence Divergence with signal line and histogram
- **ADX**: Average Directional Index for trend strength
- **Parabolic SAR**: Stop-and-reverse trend reversal points
- **DMI**: Directional Movement Index (+DI, -DI)
- **Aroon**: Trend direction and strength indicator
- **TRIX**: Rate of change of EMA for trend filtering
- **Divergence**: Price-momentum divergence signals

#### Volatility Indicators
- **Bollinger Bands**: SMA +/- 2 standard deviations envelope
- **Keltner Channels**: ATR-based envelope
- **Donchian Channels**: Highest high / lowest low channel
- **ATR Bands**: Average True Range envelope
- **Chaikin Volatility**: Accumulation/distribution volatility
- **Fibonacci Retracement**: Key support/resistance levels

#### Volume Indicators
- **OBV**: On-Balance Volume cumulative flow
- **Volume ROC**: Rate of change of volume
- **Money Flow Index**: Volume-weighted RSI
- **Chaikin Money Flow**: Accumulation/distribution flow
- **VWAP**: Volume-weighted average price

#### Moving Averages
- **SMA**: Simple Moving Average
- **EMA**: Exponential Moving Average
- **WMA**: Weighted Moving Average
- **Hull MA**: Hull Moving Average (reduced lag)
- **Triple EMA**: TEMA for ultra-low lag
- **VWMA**: Volume-Weighted Moving Average
- **VWAP**: Volume-weighted average price

#### Multi-Panel Indicators
- **Ichimoku Cloud**: Full Ichimoku with Kijun, Tenkan, Chikou, Senkou spans
- **Fibonacci Retracement**: Dynamic support/resistance levels
- **ZigZag**: Price trend reversal pattern detection
- **Pivot Points**: Standard/Fibonacci/Camarilla/Woodie pivot levels

---

## Key Algorithms

### Implied Returns Calculation

```python
def _calculate_implied_returns(self):
    # Market equilibrium returns via reverse optimization
    portfolio_variance = w_mkt' * Sigma * w_mkt
    implied_returns = delta * Sigma * w_mkt
    return implied_returns
```

### Posterior Computation (Bayesian Update)

```python
def calculate_posterior(self):
    tau_cov = tau * Sigma
    tau_cov_inv = inv(tau_cov)
    omega_inv = inv(Omega)

    # Posterior mean: combine prior with views
    A = tau_cov_inv + P' * Omega_inv * P
    A_inv = inv(A)
    B = tau_cov_inv * Pi + P' * Omega_inv * Q
    mu_bl = A_inv * B

    # Posterior covariance: increases uncertainty
    Sigma_bl = Sigma + A_inv

    return mu_bl, Sigma_bl
```

### Optimization Objective

```python
def objective(weights):
    portfolio_return = sum(w * mu_bl)
    portfolio_variance = w' * Sigma_bl * w
    return -(portfolio_return - 0.5 * delta * portfolio_variance)
```

---

## Troubleshooting

### Common Issues

#### "No data downloaded. Please check date range and tickers."

- Ensure tickers are valid Yahoo Finance symbols (try `AAPL` or `GOOGL`)
- Check that the date range has sufficient data (at least 6 months recommended)
- Some tickers may be delisted or changed; try alternative symbols

#### "Yahoo Finance request failed before price data was returned."

- This is typically caused by an outdated `yfinance` version
- Upgrade: `pip install yfinance --upgrade`
- Retry the request (Yahoo Finance APIs can be intermittent)

#### "Optimization failed: Singular matrix"

- This occurs when the covariance matrix is singular (assets with identical returns)
- Try removing one of the correlated assets
- Check for tickers with insufficient price history

#### "MemoryError" with large ticker universes

- Reduce the number of tickers (100+ can be memory-intensive)
- Use a shorter date range
- Ensure sufficient system RAM (minimum 8GB recommended)

#### Charts appear blank or empty

- Ensure all required packages are installed: `pip install -r requirements.txt`
- Clear Streamlit cache: click "Rerun" in the hamburger menu
- Check browser console for JavaScript errors

---

## Development

### Project Structure

```
ArthaDrishti-Portfolio-Studio/
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # MIT License
├── config/
│   └── settings.py                # Central configuration
├── data/
│   └── yahoo_data.py              # Yahoo Finance data fetching
├── models/
│   ├── __init__.py
│   ├── black_litterman.py         # BL model implementation
│   └── optimization.py            # Portfolio optimization
├── utils/
│   ├── __init__.py
│   ├── helpers.py                 # Utility functions
│   ├── visualization.py           # Plotly visualization engine
│   └── fix_viz.py                 # Visualization patch script
└── fonts/                         # (Optional) Custom fonts
```

### Development Workflow

1. Make changes in the appropriate module
2. Test locally with `streamlit run app.py`
3. Verify visualizations render correctly
4. Commit with descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: X - description of change"
   git push origin main
   ```

### Code Style

- Follow PEP 8 guidelines
- Use `black` for formatting: `black *.py`
- Use `ruff` for linting: `ruff check *.py`
- Keep code comments minimal but informative
- Type hints for function signatures (where practical)

---

## Deployment

### Streamlit Community Cloud

1. Push code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Click "Deploy"

### Local Deployment

```bash
pip install streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
```

```bash
docker build -t arthadrishti-app .
docker run -p 8501:8501 arthadrishti-app
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Author**: Sourish Dey  
**Roll No**: 23051223  
**Program**: Bachelors of Technology in Computer Science  
**University**: KIIT University, Bhubaneswar, Odisha  
**Email**: 23051223@kiit.ac.in  

---

## References

1. Black, F. and Litterman, R. (1992). "Global Portfolio Optimization." *Goldman Sachs Investment Research*.
2. He, P. and Litterman, R. (2002). "The Intuitive Spectral Approach to Black-Litterman." *Goldman Sachs*.
3. Markowitz, H. (1952). "Portfolio Selection." *The Journal of Finance*, 7(1), 77-91.
4. Sharpe, W.F. (1970). "Asset Valuation and Portfolio Selection." *Journal of Finance*.
5. [yfinance Documentation](https://pypi.org/project/yfinance/)
6. [Plotly Python Documentation](https://plotly.com/python/)
7. [Streamlit Documentation](https://docs.streamlit.io/)
8. [SciPy Optimization Documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html)
9. [PyPortfolioOpt Documentation](https://pyportfolioopt.readthedocs.io/)
10. [Black-Litterman on Wikipedia](https://en.wikipedia.org/wiki/Black%E2%80%93Litterman_model)
