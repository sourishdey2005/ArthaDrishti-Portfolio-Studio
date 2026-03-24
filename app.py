"""
Main Streamlit Application for the ArthaDrishti portfolio studio.
"""
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import *
from data.yahoo_data import YahooFinanceData
from models.black_litterman import BlackLittermanModel
from models.optimization import PortfolioOptimizer
from utils.helpers import format_percentage
from utils.visualization import Visualizer

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="ArthaDrishti Portfolio Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
    /* Main App Background - Professional Dark Theme */
    .stApp {
        background:
            radial-gradient(circle at 20% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(16, 185, 129, 0.12) 0%, transparent 50%),
            radial-gradient(circle at top right, rgba(255, 153, 51, 0.1), transparent 34%),
            radial-gradient(circle at top left, rgba(19, 136, 8, 0.08), transparent 30%),
            linear-gradient(180deg, #0a0e17 0%, #0f1729 40%, #111827 100%);
        color: #f1f5f9;
    }
    
    /* App View Container */
    [data-testid="stAppViewContainer"] {
        background: transparent;
    }
    
    /* Header Styling */
    [data-testid="stHeader"] {
        background: rgba(10, 14, 23, 0.85);
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Sidebar - Professional Dark */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e17 0%, #0f1729 50%, #111827 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    
    /* Sidebar Section Headers */
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4 {
        color: #818cf8 !important;
        font-weight: 600;
        letter-spacing: 0.025em;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        color: #f8fafc;
        font-weight: 700;
        font-size: 1.5rem;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-weight: 500;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 500;
        padding: 0.75rem 1.25rem;
        border-radius: 8px 8px 0 0;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc;
        background: rgba(99, 102, 241, 0.1);
    }
    .stTabs [aria-selected="true"] {
        color: #818cf8 !important;
        background: rgba(99, 102, 241, 0.15) !important;
        border-bottom: 2px solid #818cf8;
    }
    
    /* DataFrame and Table */
    .stDataFrame, .stTable {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.15);
    }
    
    /* Buttons - Professional Styling */
    .stButton > button {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.8) 0%, rgba(79, 70, 229, 0.9) 100%);
        color: #f8fafc;
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 10px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(129, 140, 248, 0.9) 0%, rgba(99, 102, 241, 1) 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
    }
    .stButton > button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(51, 65, 85, 0.9);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > div,
    .stSelectbox > div > div > div {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        color: #f8fafc;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #818cf8;
        box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.15);
    }
    
    /* Slider Styling */
    .stSlider [data-baseweb="slider"] {
        background: rgba(51, 65, 85, 0.5);
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #818cf8;
        border: 2px solid #6366f1;
    }
    
    /* Checkbox */
    .stCheckbox > label > div:first-child {
        border-color: #818cf8;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f8fafc 0%, #818cf8 50%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        letter-spacing: 0.02em;
        text-shadow: 0 2px 20px rgba(129, 140, 248, 0.3);
    }
    
    /* Brand Caption */
    .brand-caption {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    
    /* Sub Header */
    .sub-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 1.25rem;
        margin-bottom: 1.25rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(129, 140, 248, 0.25);
    }
    
    /* Success Message */
    .success-message {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.1) 100%);
        color: #d1fae5;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
    }
    
    /* Warning Message */
    .warning-message {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.1) 100%);
        color: #fef3c7;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid rgba(245, 158, 11, 0.3);
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(14, 165, 233, 0.08) 100%);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #38bdf8;
        color: #e0f2fe;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.1);
    }
    
    /* Footer Credit */
    .footer-credit {
        margin-top: 3rem;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.15) 0%, rgba(16, 185, 129, 0.12) 50%, rgba(99, 102, 241, 0.15) 100%);
        color: #f8fafc;
        font-weight: 600;
        border: 1px solid rgba(129, 140, 248, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .footer-credit a {
        color: #818cf8;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    .footer-credit a:hover {
        color: #a5b4fc;
        text-decoration: underline;
    }
    
    /* Divider */
    hr {
        border-color: rgba(99, 102, 241, 0.2);
        margin: 1.5rem 0;
    }
    
    /* Expanders */
    .streamlit-expander {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.15);
    }
    
    /* Radio Buttons */
    .stRadio > label > div {
        color: #e2e8f0;
    }
    
    /* Multiselect */
    .stMultiSelect [data-baseweb="multiselect"] {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
    }
    
    /* Tooltip */
    [data-testid="stTooltipContent"] {
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
    }
    
    /* Spinner */
    [data-testid="stSpinner"] {
        color: #818cf8;
    }
    
    /* Progress Bar */
    [data-testid="stProgressBar"] > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #818cf8 100%);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.4);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.6);
    }
    
    /* Chart Container Enhancement */
    [data-testid="stPlotlyChart"] {
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.1);
        background: rgba(15, 23, 42, 0.5);
        padding: 0.5rem;
    }
    
    /* Section Headers in Sidebar */
    [data-testid="stSidebar"] .stMarkdown p {
        color: #cbd5e1;
    }
    
    /* Quick Select Button Group */
    [data-testid="stSidebar"] .stButton > button {
        font-size: 0.85rem;
        padding: 0.5rem 0.75rem;
    }
    
    /* Tab Container */
    [data-testid="stTabPane"] {
        padding: 1rem 0;
    }
    
    /* Streamlit Alert */
    .stAlert {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.8) 0%, rgba(5, 150, 105, 0.9) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.9) 0%, rgba(16, 185, 129, 1) 100%);
    }
</style>
""",
    unsafe_allow_html=True
)


for key, default_value in {
    "bl_model": None,
    "views_added": [],
    "results": None,
    "returns": None,
    "market_caps": None,
    "tickers": [],
    "selected_tickers": DEFAULT_TICKERS.copy(),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


def render_chart_pairs(charts):
    for index in range(0, len(charts), 2):
        col1, col2 = st.columns(2)
        title1, fig1 = charts[index]
        with col1:
            st.markdown(f"#### {title1}")
            st.plotly_chart(fig1, use_container_width=True)
        if index + 1 < len(charts):
            title2, fig2 = charts[index + 1]
            with col2:
                st.markdown(f"#### {title2}")
                st.plotly_chart(fig2, use_container_width=True)


def build_analysis_bundle(
    bl_model,
    tickers,
    start_date_input,
    end_date_input,
    allow_short,
    max_weight,
    risk_free_rate
):
    posterior_returns, posterior_cov = bl_model.calculate_posterior()
    prior_returns = bl_model.pi
    optimal_weights = bl_model.get_optimal_weights(
        allow_short=allow_short,
        max_weight=max_weight
    )
    optimizer = PortfolioOptimizer(
        expected_returns=posterior_returns,
        cov_matrix=posterior_cov,
        risk_free_rate=risk_free_rate
    )
    portfolio_stats = optimizer.calculate_portfolio_stats(optimal_weights)
    target_returns, frontier_volatilities, frontier_weights = optimizer.get_efficient_frontier(
        n_points=50,
        allow_short=allow_short,
        max_weight=max_weight
    )
    results_df = bl_model.get_results_dataframe()

    price_fetcher = YahooFinanceData(
        tickers=tickers,
        start_date=start_date_input.strftime("%Y-%m-%d"),
        end_date=end_date_input.strftime("%Y-%m-%d")
    )
    normalized_prices = price_fetcher.get_price_history()
    ohlc_history = price_fetcher.get_ohlc_history()

    return {
        "posterior_returns": posterior_returns,
        "posterior_cov": posterior_cov,
        "prior_returns": prior_returns,
        "optimal_weights": optimal_weights,
        "optimizer": optimizer,
        "portfolio_stats": portfolio_stats,
        "target_returns": target_returns,
        "frontier_volatilities": frontier_volatilities,
        "frontier_weights": frontier_weights,
        "results_df": results_df,
        "normalized_prices": normalized_prices,
        "ohlc_history": ohlc_history,
    }


def main():
    st.markdown('<div class="main-header">ArthaDrishti Portfolio Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-caption">An Indian-inspired Black-Litterman workspace for market views, portfolio signals, and visual analytics.</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    with st.sidebar:
        st.markdown("### 🎯 Portfolio Settings")
        st.markdown("#### 📊 Asset Selection")

        selected_tickers = st.multiselect(
            "Select stock tickers from a 100-stock dropdown",
            options=TICKER_UNIVERSE_100,
            default=st.session_state.selected_tickers,
            help="Choose one or more tickers from the dropdown list."
        )
        st.session_state.selected_tickers = selected_tickers

        st.markdown("#### Quick Select")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📱 Tech", use_container_width=True):
                st.session_state.selected_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA"]
                st.rerun()
        with col2:
            if st.button("🏦 Finance", use_container_width=True):
                st.session_state.selected_tickers = ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB"]
                st.rerun()
        with col3:
            if st.button("💊 Health", use_container_width=True):
                st.session_state.selected_tickers = ["JNJ", "PFE", "MRK", "ABBV", "UNH", "CVS"]
                st.rerun()

        st.markdown("#### 📅 Date Range")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3 * 365)

        start_date_input = st.date_input("Start Date", value=start_date, max_value=end_date)
        end_date_input = st.date_input("End Date", value=end_date, max_value=end_date)

        st.markdown("#### ⚙️ Model Parameters")
        delta = st.slider(
            "Risk Aversion Coefficient (δ)",
            min_value=1.0,
            max_value=5.0,
            value=DELTA,
            step=0.1
        )
        tau = st.slider(
            "Prior Uncertainty (τ)",
            min_value=0.01,
            max_value=0.05,
            value=TAU,
            step=0.005
        )
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=5.0,
            value=RISK_FREE_RATE * 100,
            step=0.5
        ) / 100

        st.markdown("#### 🔒 Optimization Constraints")
        allow_short = st.checkbox("Allow Short Selling", value=ALLOW_SHORT)
        max_weight = st.slider(
            "Maximum Weight per Asset (%)",
            min_value=10,
            max_value=100,
            value=int(MAX_WEIGHT * 100),
            step=5
        ) / 100

        st.markdown("---")
        fetch_data = st.button("🚀 Fetch Data & Run Model", type="primary", use_container_width=True)

    if fetch_data:
        with st.spinner("Fetching data from Yahoo Finance..."):
            try:
                tickers = [ticker.strip().upper() for ticker in st.session_state.selected_tickers if ticker.strip()]
                if not tickers:
                    st.error("Please enter at least one valid ticker symbol")
                    st.stop()

                st.info(f"Fetching data for: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")

                data_fetcher = YahooFinanceData(
                    tickers=tickers,
                    start_date=start_date_input.strftime("%Y-%m-%d"),
                    end_date=end_date_input.strftime("%Y-%m-%d")
                )
                returns, market_caps = data_fetcher.fetch_data()

                if returns is None or returns.empty:
                    st.error("No valid data returned. Please check ticker symbols and date range.")
                    st.stop()

                if len(returns) < 30:
                    st.warning(f"Only {len(returns)} data points are available. A longer range will give stronger results.")

                bl_model = BlackLittermanModel(
                    returns=returns,
                    market_caps=market_caps,
                    delta=delta,
                    tau=tau
                )

                st.session_state.bl_model = bl_model
                st.session_state.returns = returns
                st.session_state.market_caps = market_caps
                st.session_state.tickers = data_fetcher.valid_tickers

                st.markdown(
                    f"""
                    <div class="success-message">
                    <strong>Data fetched successfully.</strong><br>
                    Found data for {len(data_fetcher.valid_tickers)} assets across {len(returns)} trading days.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if len(data_fetcher.valid_tickers) < len(tickers):
                    invalid = set(tickers) - set(data_fetcher.valid_tickers)
                    st.markdown(
                        f"""
                        <div class="warning-message">
                        <strong>Could not fetch data for:</strong> {', '.join(invalid)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.markdown(
                    """
                    <div class="info-box">
                    <strong>Troubleshooting tips:</strong><br>
                    • Check ticker symbols<br>
                    • Try a longer date range<br>
                    • Ensure internet access is available<br>
                    • Upgrade yfinance if Yahoo responses look malformed
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.stop()

    if st.session_state.bl_model is not None and st.session_state.returns is not None:
        visualizer = Visualizer()
        tickers = st.session_state.tickers
        returns_df = st.session_state.returns
        bl_model = st.session_state.bl_model

        try:
            analysis = build_analysis_bundle(
                bl_model,
                tickers,
                start_date_input,
                end_date_input,
                allow_short,
                max_weight,
                risk_free_rate
            )
        except Exception as e:
            st.error(f"Error preparing analytics: {str(e)}")
            st.stop()

        posterior_returns = analysis["posterior_returns"]
        posterior_cov = analysis["posterior_cov"]
        prior_returns = analysis["prior_returns"]
        optimal_weights = analysis["optimal_weights"]
        optimizer = analysis["optimizer"]
        portfolio_stats = analysis["portfolio_stats"]
        target_returns = analysis["target_returns"]
        frontier_volatilities = analysis["frontier_volatilities"]
        frontier_weights = analysis["frontier_weights"]
        results_df = analysis["results_df"]
        normalized_prices = analysis["normalized_prices"]
        ohlc_history = analysis["ohlc_history"]

        market_weights = pd.Series(bl_model.market_weights, index=tickers)
        optimal_weights_series = pd.Series(optimal_weights, index=tickers)
        valid_frontier = ~np.isnan(frontier_volatilities)
        lead_ticker = tickers[0] if tickers else None
        lead_ohlc = ohlc_history.get(lead_ticker) if lead_ticker else None

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Overview",
            "🎯 Investor Views",
            "📈 Results",
            "⚡ Optimization"
        ])

        with tab1:
            st.markdown('<div class="sub-header">Data Overview</div>', unsafe_allow_html=True)
            data_tabs = st.tabs(["Price Action", "Seasonality", "Distribution", "Structure"])

            with data_tabs[0]:
                charts = []
                if normalized_prices is not None and not normalized_prices.empty:
                    charts.extend([
                        ("Normalized Price History", visualizer.plot_price_history(normalized_prices)),
                        ("Cumulative Returns", visualizer.plot_cumulative_returns(returns_df)),
                        ("Drawdown Curves", visualizer.plot_drawdown_chart(returns_df)),
                    ])
                if lead_ohlc is not None and not lead_ohlc.empty:
                    charts.extend([
                        (f"Candlestick: {lead_ticker}", visualizer.plot_candlestick_chart(lead_ohlc, lead_ticker)),
                        (f"Candlestick + Volume: {lead_ticker}", visualizer.plot_candlestick_with_volume(lead_ohlc, lead_ticker)),
                        (f"OHLC: {lead_ticker}", visualizer.plot_ohlc_chart(lead_ohlc, lead_ticker)),
                        (f"High Low Band: {lead_ticker}", visualizer.plot_high_low_band(lead_ohlc, lead_ticker)),
                        (f"Daily Range: {lead_ticker}", visualizer.plot_daily_range_bar(lead_ohlc, lead_ticker)),
                        (f"Opening Gaps: {lead_ticker}", visualizer.plot_gap_bar(lead_ohlc, lead_ticker)),
                        (f"Close vs Volume: {lead_ticker}", visualizer.plot_close_vs_volume_scatter(lead_ohlc, lead_ticker)),
                        (f"Volume Bars: {lead_ticker}", visualizer.plot_volume_bars(lead_ohlc, lead_ticker)),
                        # NEW Technical Analysis Visualizations
                        (f"RSI: {lead_ticker}", visualizer.plot_rsi_indicator(lead_ohlc, lead_ticker)),
                        (f"MACD: {lead_ticker}", visualizer.plot_macd_indicator(lead_ohlc, lead_ticker)),
                        (f"Bollinger Bands: {lead_ticker}", visualizer.plot_bollinger_bands(lead_ohlc, lead_ticker)),
                        (f"Stochastic: {lead_ticker}", visualizer.plot_stochastic_oscillator(lead_ohlc, lead_ticker)),
                        (f"On Balance Volume: {lead_ticker}", visualizer.plot_on_balance_volume(lead_ohlc, lead_ticker)),
                        (f"ATR: {lead_ticker}", visualizer.plot_average_true_range(lead_ohlc, lead_ticker)),
                        (f"MFI: {lead_ticker}", visualizer.plot_money_flow_index(lead_ohlc, lead_ticker)),
                        (f"Williams %R: {lead_ticker}", visualizer.plot_williams_r(lead_ohlc, lead_ticker)),
                        (f"CCI: {lead_ticker}", visualizer.plot_commodity_channel_index(lead_ohlc, lead_ticker)),
                        (f"ROC: {lead_ticker}", visualizer.plot_price_rate_of_change(lead_ohlc, lead_ticker)),
                        (f"TRIX: {lead_ticker}", visualizer.plot_trix_indicator(lead_ohlc, lead_ticker)),
                        (f"Donchian Channels: {lead_ticker}", visualizer.plot_donchian_channels(lead_ohlc, lead_ticker)),
                        (f"VWAP: {lead_ticker}", visualizer.plot_vwap_indicator(lead_ohlc, lead_ticker)),
                        (f"Ichimoku Cloud: {lead_ticker}", visualizer.plot_ichimoku_cloud(lead_ohlc, lead_ticker)),
                        (f"Pivot Points: {lead_ticker}", visualizer.plot_pivot_points(lead_ohlc, lead_ticker)),
                        (f"Ease of Movement: {lead_ticker}", visualizer.plot_ease_of_movement(lead_ohlc, lead_ticker)),
                        (f"Force Index: {lead_ticker}", visualizer.plot_force_index(lead_ohlc, lead_ticker)),
                        (f"Ultimate Oscillator: {lead_ticker}", visualizer.plot_ultimate_oscillator(lead_ohlc, lead_ticker)),
                        (f"Aroon: {lead_ticker}", visualizer.plot_aroon_indicator(lead_ohlc, lead_ticker)),
                        (f"Chaikin Oscillator: {lead_ticker}", visualizer.plot_chaikin_oscillator(lead_ohlc, lead_ticker)),
                        (f"Demand Index: {lead_ticker}", visualizer.plot_demand_index(lead_ohlc, lead_ticker)),
                        (f"Mass Index: {lead_ticker}", visualizer.plot_mass_index(lead_ohlc, lead_ticker)),
                        (f"TSI: {lead_ticker}", visualizer.plot_true_strength_index(lead_ohlc, lead_ticker)),
                        (f"KST: {lead_ticker}", visualizer.plot_kst_indicator(lead_ohlc, lead_ticker)),
                        (f"A/D Line: {lead_ticker}", visualizer.plot_accumulation_distribution(lead_ohlc, lead_ticker)),
                        (f"Volume Profile: {lead_ticker}", visualizer.plot_volume_profile(lead_ohlc, lead_ticker)),
                        (f"PVT: {lead_ticker}", visualizer.plot_price_volume_trend(lead_ohlc, lead_ticker)),
                        (f"Heikin Ashi: {lead_ticker}", visualizer.plot_heikin_ashi_candles(lead_ohlc, lead_ticker)),
                        (f"Keltner Channels: {lead_ticker}", visualizer.plot_keltner_channels(lead_ohlc, lead_ticker)),
                        (f"Parabolic SAR: {lead_ticker}", visualizer.plot_parabolic_sar(lead_ohlc, lead_ticker)),
                    ])
                render_chart_pairs(charts)

            with data_tabs[1]:
                render_chart_pairs([
                    ("Monthly Heatmap", visualizer.plot_monthly_returns_heatmap(returns_df)),
                    ("Weekday Seasonality", visualizer.plot_return_heatmap_by_weekday(returns_df)),
                    ("Calendar Return Bars", visualizer.plot_calendar_return_bars(returns_df)),
                    ("Annual Returns", visualizer.plot_annual_returns_bar(returns_df)),
                    ("Quarterly Returns", visualizer.plot_quarterly_returns_bar(returns_df)),
                    ("Monthly Return Distribution", visualizer.plot_monthly_boxplot(returns_df)),
                ])

            with data_tabs[2]:
                render_chart_pairs([
                    ("Returns Distribution", visualizer.plot_returns_distribution(returns_df)),
                    ("Return Boxplot", visualizer.plot_return_boxplot(returns_df)),
                    ("Return Violin Plot", visualizer.plot_return_violin(returns_df)),
                    ("Return Quantile Band", visualizer.plot_return_quantile_band(returns_df)),
                    ("Best vs Worst Day", visualizer.plot_best_worst_days(returns_df)),
                    ("Autocorrelation", visualizer.plot_return_autocorrelation(returns_df)),
                    ("Lag Scatter", visualizer.plot_lag_scatter(returns_df)),
                    ("Return Polar Chart", visualizer.plot_return_polar_bar(returns_df)),
                ])

            with data_tabs[3]:
                summary_stats = returns_df.describe().T
                summary_stats["Mean Return (%)"] = summary_stats["mean"] * 100
                summary_stats["Std Dev (%)"] = summary_stats["std"] * 100
                summary_stats["Sharpe (Ann.)"] = (summary_stats["mean"] / summary_stats["std"]) * np.sqrt(252)
                st.dataframe(summary_stats[["Mean Return (%)", "Std Dev (%)", "Sharpe (Ann.)"]].round(2), use_container_width=True)
                render_chart_pairs([
                    ("Correlation Heatmap", visualizer.plot_correlation_heatmap(returns_df.corr())),
                    ("Correlation Bubble", visualizer.plot_correlation_bubble(returns_df.corr())),
                    ("Covariance Heatmap", visualizer.plot_covariance_heatmap(posterior_cov, tickers)),
                    ("Pairwise Beta Heatmap", visualizer.plot_beta_heatmap(returns_df)),
                    ("Scatter Matrix", visualizer.plot_returns_scatter_matrix(returns_df)),
                ])

        with tab2:
            st.markdown('<div class="sub-header">Investor Views</div>', unsafe_allow_html=True)

            view_type = st.radio("Select view type", ["Absolute View", "Relative View"], horizontal=True)
            if view_type == "Absolute View":
                col1, col2, col3 = st.columns(3)
                with col1:
                    asset = st.selectbox("Asset", tickers)
                with col2:
                    expected_return = st.number_input("Expected Return (%)", value=10.0, step=0.5, format="%.1f") / 100
                with col3:
                    confidence = st.slider("Confidence Level", min_value=0.1, max_value=1.0, value=0.7, step=0.05)
                if st.button("Add Absolute View", use_container_width=True):
                    bl_model.add_absolute_view(asset=asset, expected_return=expected_return, confidence=confidence)
                    st.session_state.views_added.append(f"Absolute: {asset} -> {expected_return*100:.1f}%")
                    st.success(f"Added a view for {asset}.")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    asset_out = st.selectbox("Outperform Asset", tickers, key="out")
                with col2:
                    asset_under = st.selectbox("Underperform Asset", tickers, key="under")
                with col3:
                    expected_outperformance = st.number_input("Expected Outperformance (%)", value=5.0, step=0.5, format="%.1f") / 100
                with col4:
                    confidence_rel = st.slider("Confidence Level", min_value=0.1, max_value=1.0, value=0.7, step=0.05, key="conf_rel")
                if st.button("Add Relative View", use_container_width=True):
                    if asset_out == asset_under:
                        st.error("Please choose different assets.")
                    else:
                        bl_model.add_relative_view(
                            asset_outperform=asset_out,
                            asset_underperform=asset_under,
                            expected_outperformance=expected_outperformance,
                            confidence=confidence_rel
                        )
                        st.session_state.views_added.append(
                            f"Relative: {asset_out} over {asset_under} by {expected_outperformance*100:.1f}%"
                        )
                        st.success("Relative view added.")

            if st.session_state.views_added:
                st.markdown("#### Current Views")
                for view in st.session_state.views_added:
                    st.markdown(f"- {view}")
                if st.button("Clear All Views", use_container_width=True):
                    st.session_state.views_added = []
                    bl_model.P = None
                    bl_model.Q = None
                    bl_model.views_uncertainty = None
                    st.success("All views cleared.")
                    st.rerun()
            else:
                st.info("No views added yet.")

            view_tabs = st.tabs(["View Impact", "Weight Shifts", "Conviction Lens"])
            with view_tabs[0]:
                render_chart_pairs([
                    ("Prior vs Posterior Returns", visualizer.plot_returns_comparison(prior_returns, posterior_returns, tickers)),
                    ("Weight vs Expected Return", visualizer.plot_weight_vs_return_scatter(optimal_weights, posterior_returns, tickers)),
                    ("Radar Performance", visualizer.plot_radar_performance(returns_df, risk_free_rate)),
                    ("Parallel Coordinates", visualizer.plot_parallel_coordinates(returns_df, risk_free_rate)),
                ])
            with view_tabs[1]:
                render_chart_pairs([
                    ("Active Weights", visualizer.plot_active_weight_bar(market_weights, optimal_weights_series, tickers)),
                    ("Market to Optimal Flow", visualizer.plot_sankey_allocation_flow(market_weights, optimal_weights_series, tickers)),
                    ("Allocation Sunburst", visualizer.plot_sunburst_allocation(optimal_weights, posterior_returns, tickers)),
                    ("Weight Bubble Map", visualizer.plot_weight_bubble_chart(optimal_weights, posterior_returns, posterior_cov, tickers)),
                    ("Allocation Polar View", visualizer.plot_concentration_polar(optimal_weights, tickers)),
                    ("Weight Comparison", visualizer.plot_weights_comparison(
                        market_weights * 100,
                        pd.Series(prior_returns * 100, index=tickers),
                        optimal_weights_series * 100
                    )),
                ])
            with view_tabs[2]:
                render_chart_pairs([
                    ("Mean Returns", visualizer.plot_mean_returns_bar(returns_df)),
                    ("Sharpe Ratios", visualizer.plot_sharpe_bar(returns_df, risk_free_rate)),
                    ("Sortino Ratios", visualizer.plot_sortino_bar(returns_df, risk_free_rate)),
                    ("Return Ranking", visualizer.plot_return_rank_bar(returns_df)),
                    ("Volatility Ranking", visualizer.plot_volatility_rank_bar(returns_df)),
                    ("Sharpe Ranking", visualizer.plot_sharpe_rank_bar(returns_df, risk_free_rate)),
                ])

        with tab3:
            st.markdown('<div class="sub-header">Results</div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Number of Assets", len(tickers))
            with col2:
                st.metric("Number of Views", len(st.session_state.views_added))
            with col3:
                st.metric("Avg Market Weight", f"{results_df['Market Weight'].mean():.1f}%")
            with col4:
                st.metric("Avg Optimal Weight", f"{results_df['Optimal Weight (%)'].mean():.1f}%")

            result_tabs = st.tabs(["Summary", "Risk Lens", "Return Lens"])
            with result_tabs[0]:
                render_chart_pairs([
                    ("Prior vs Posterior Returns", visualizer.plot_returns_comparison(prior_returns, posterior_returns, tickers)),
                    ("Active Weights", visualizer.plot_active_weight_bar(market_weights, optimal_weights_series, tickers)),
                    ("Return Contribution", visualizer.plot_return_contribution_bar(returns_df, optimal_weights, tickers)),
                    ("3D Risk Return Weight", visualizer.plot_3d_risk_return_weights(returns_df, optimal_weights, risk_free_rate)),
                ])
                st.dataframe(results_df.round(2), use_container_width=True, height=400)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"artha_drishti_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with result_tabs[1]:
                render_chart_pairs([
                    ("Volatility Bars", visualizer.plot_volatility_bar(returns_df)),
                    ("Value at Risk", visualizer.plot_var_bar(returns_df)),
                    ("Conditional VaR", visualizer.plot_cvar_bar(returns_df)),
                    ("Tail Risk Bubble", visualizer.plot_tail_risk_bubble(returns_df)),
                    ("Max Drawdown", visualizer.plot_max_drawdown_bar(returns_df)),
                    ("Drawdown Duration", visualizer.plot_drawdown_duration_bar(returns_df)),
                    ("Underwater Heatmap", visualizer.plot_underwater_heatmap(returns_df)),
                    ("Rolling Beta", visualizer.plot_rolling_beta(returns_df)),
                ])
            with result_tabs[2]:
                render_chart_pairs([
                    ("Positive Days", visualizer.plot_positive_days_bar(returns_df)),
                    ("Negative Days", visualizer.plot_negative_days_bar(returns_df)),
                    ("Expanding Sharpe", visualizer.plot_expanding_sharpe(returns_df, risk_free_rate)),
                    ("Rolling Skewness", visualizer.plot_rolling_skewness(returns_df)),
                    ("Rolling Kurtosis", visualizer.plot_rolling_kurtosis(returns_df)),
                    ("Regime Strip", visualizer.plot_regime_strip(returns_df)),
                ])

        with tab4:
            st.markdown('<div class="sub-header">Optimization</div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Expected Annual Return", format_percentage(portfolio_stats["return"] * 100))
            with col2:
                st.metric("Annual Volatility", format_percentage(portfolio_stats["volatility"] * 100))
            with col3:
                st.metric("Sharpe Ratio", f"{portfolio_stats['sharpe_ratio']:.2f}")
            with col4:
                st.metric("Diversification Ratio", f"{portfolio_stats['diversification_ratio']:.2f}")

            optimization_tabs = st.tabs(["Frontier", "Allocation", "Risk Engine", "Advanced 3D"])
            with optimization_tabs[0]:
                render_chart_pairs([
                    ("Efficient Frontier", visualizer.plot_efficient_frontier(
                        target_returns[valid_frontier],
                        frontier_volatilities[valid_frontier],
                        portfolio_stats
                    )),
                    ("Frontier Return Distribution", visualizer.plot_frontier_return_distribution(target_returns[valid_frontier])),
                    ("Frontier Sharpe Profile", visualizer.plot_frontier_sharpe_area(
                        target_returns[valid_frontier],
                        frontier_volatilities[valid_frontier],
                        risk_free_rate
                    )),
                    ("Frontier Weights Heatmap", visualizer.plot_frontier_weights_heatmap(frontier_weights, tickers)),
                    ("Relative Performance", visualizer.plot_cumulative_relative_performance(returns_df, optimal_weights)),
                    ("Portfolio Growth", visualizer.plot_portfolio_growth(returns_df, optimal_weights)),
                ])
            with optimization_tabs[1]:
                allocation_dict = {
                    ticker: weight * 100
                    for ticker, weight in optimal_weights_series.items()
                    if weight > 0.0001
                }
                render_chart_pairs([
                    ("Portfolio Allocation", visualizer.plot_portfolio_allocation(allocation_dict, title="Optimal Portfolio Allocation")),
                    ("Allocation Treemap", visualizer.plot_weight_treemap(optimal_weights_series * 100)),
                    ("Allocation Waterfall", visualizer.plot_weight_waterfall(optimal_weights_series)),
                    ("Active Weights", visualizer.plot_active_weight_bar(market_weights, optimal_weights_series, tickers)),
                    ("Allocation Flow", visualizer.plot_sankey_allocation_flow(market_weights, optimal_weights_series, tickers)),
                    ("Allocation Polar View", visualizer.plot_concentration_polar(optimal_weights, tickers)),
                ])
            with optimization_tabs[2]:
                render_chart_pairs([
                    ("Risk Contribution", visualizer.plot_risk_contribution_bar(optimal_weights, posterior_cov, tickers)),
                    ("Risk Contribution Treemap", visualizer.plot_risk_contribution_treemap(optimal_weights, posterior_cov, tickers)),
                    ("Diversification Benefit", visualizer.plot_diversification_benefit(returns_df, optimal_weights, posterior_cov, tickers)),
                    ("Rolling Correlation", visualizer.plot_rolling_correlation(returns_df)),
                    ("Rolling Beta", visualizer.plot_rolling_beta(returns_df)),
                    ("Return Contribution", visualizer.plot_return_contribution_bar(returns_df, optimal_weights, tickers)),
                ])
            with optimization_tabs[3]:
                render_chart_pairs([
                    ("3D Efficient Frontier", visualizer.plot_3d_frontier(
                        target_returns[valid_frontier],
                        frontier_volatilities[valid_frontier],
                        [frontier_weights[index] for index in range(len(frontier_weights)) if valid_frontier[index]],
                        tickers
                    )),
                    ("3D Covariance Surface", visualizer.plot_3d_covariance_surface(posterior_cov, tickers)),
                    ("3D Risk Return Weight", visualizer.plot_3d_risk_return_weights(returns_df, optimal_weights, risk_free_rate)),
                ])

    st.markdown(
        '<div class="footer-credit">Made By Sourish Dey , <a href="https://sourishdeyportfolio.vercel.app/" target="_blank">https://sourishdeyportfolio.vercel.app/</a></div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
