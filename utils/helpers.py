"""
Helper functions for the Black-Litterman Model application
"""
import pandas as pd
import numpy as np


def format_currency(value, currency='USD'):
    """
    Format number as currency
    """
    if pd.isna(value):
        return 'N/A'
    
    if currency == 'USD':
        return f'${value:,.2f}'
    else:
        return f'{value:,.2f}'


def format_percentage(value, decimals=2):
    """
    Format number as percentage
    """
    if pd.isna(value):
        return 'N/A'
    
    return f'{value:.{decimals}f}%'


def calculate_portfolio_metrics(weights, expected_returns, cov_matrix, risk_free_rate=0.02):
    """
    Calculate portfolio performance metrics
    """
    # Calculate portfolio return
    portfolio_return = np.sum(weights * expected_returns)
    
    # Calculate portfolio volatility
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Calculate Sharpe ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Calculate diversification ratio
    weighted_vols = np.sum(weights * np.sqrt(np.diag(cov_matrix)))
    diversification_ratio = weighted_vols / portfolio_volatility
    
    return {
        'return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio,
        'diversification_ratio': diversification_ratio
    }


def validate_tickers(tickers):
    """
    Validate and clean ticker symbols
    """
    # Remove duplicates and empty strings
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))
    
    # Remove any non-alphanumeric characters except dot
    tickers = [''.join(c for c in t if c.isalnum() or c == '.') for t in tickers]
    
    return tickers


def calculate_market_cap_weights(market_caps):
    """
    Calculate market capitalization weights
    """
    if isinstance(market_caps, pd.Series):
        return market_caps / market_caps.sum()
    elif isinstance(market_caps, dict):
        total = sum(market_caps.values())
        return {k: v / total for k, v in market_caps.items()}
    else:
        raise ValueError("market_caps must be dict or Series")


def calculate_historical_volatility(returns, annualize=True):
    """
    Calculate historical volatility
    """
    volatility = returns.std()
    
    if annualize:
        volatility = volatility * np.sqrt(252)
    
    return volatility


def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown from price series
    """
    rolling_max = prices.expanding().max()
    drawdown = (prices - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return max_drawdown


def calculate_var(returns, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR)
    """
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[index]
    
    return var


def calculate_cvar(returns, confidence_level=0.95):
    """
    Calculate Conditional Value at Risk (CVaR)
    """
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    cvar = np.mean(sorted_returns[:index])
    
    return cvar


def calculate_turnover(weights_new, weights_old):
    """
    Calculate portfolio turnover
    """
    turnover = np.sum(np.abs(weights_new - weights_old)) / 2
    
    return turnover


def create_summary_table(results_df):
    """
    Create a formatted summary table
    """
    summary = pd.DataFrame({
        'Metric': ['Total Return', 'Volatility', 'Sharpe Ratio', 'Diversification Ratio'],
        'Value': [
            results_df['Return'].sum(),
            results_df['Volatility'].mean(),
            results_df['Sharpe Ratio'].mean(),
            results_df['Diversification Ratio'].mean()
        ]
    })
    
    return summary