"""
Configuration settings for the Black-Litterman Model application
"""

# Market settings
RISK_FREE_RATE = 0.02  # 2% risk-free rate
TAU = 0.025  # Uncertainty in the prior (typically 0.025-0.05)
DELTA = 2.5  # Risk aversion coefficient (typically 2-4)

# Default assets for demonstration
DEFAULT_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'JPM', 'V', 'JNJ', 'WMT', 'PG']
TICKER_UNIVERSE_100 = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'ORCL',
    'CRM', 'ADBE', 'INTC', 'CSCO', 'IBM', 'QCOM', 'AVGO', 'TXN', 'AMAT', 'MU',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'AXP', 'SCHW', 'BLK',
    'JNJ', 'PFE', 'MRK', 'ABBV', 'UNH', 'CVS', 'LLY', 'TMO', 'ABT', 'DHR',
    'WMT', 'COST', 'HD', 'LOW', 'TGT', 'NKE', 'MCD', 'SBUX', 'KO', 'PEP',
    'PG', 'CL', 'KMB', 'EL', 'MDLZ', 'GIS', 'KHC', 'MO', 'PM', 'ADM',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI',
    'BA', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX',
    'DIS', 'CMCSA', 'TMUS', 'VZ', 'T', 'SPGI', 'MOODY', 'PYPL', 'SQ', 'UBER',
    'V', 'MA', 'INTU', 'BKNG', 'ISRG', 'ADI', 'LRCX', 'PANW', 'SNOW', 'SHOP'
]

# Date range for historical data
DEFAULT_START_DATE = '2020-01-01'
DEFAULT_END_DATE = '2023-12-31'

# Optimization constraints
ALLOW_SHORT = False
MAX_WEIGHT = 0.3  # Maximum weight per asset (30%)
MIN_WEIGHT = 0.0  # Minimum weight per asset

# Visualization settings
THEME_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'background': "#04305d",
    'text': '#2c3e50'
}
