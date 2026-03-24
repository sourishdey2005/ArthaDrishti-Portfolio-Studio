"""
Yahoo Finance Data Fetching Module
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class YahooFinanceData:
    """
    Fetch and process financial data from Yahoo Finance
    """
    
    def __init__(self, tickers, start_date, end_date):
        """
        Initialize data fetcher
        
        Parameters:
        -----------
        tickers : list
            List of stock tickers
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.market_caps = None
        self.valid_tickers = []

    @staticmethod
    def _extract_close_prices(data, tickers):
        """
        Normalize yfinance download output into a prices DataFrame.
        """
        if data is None or data.empty:
            return pd.DataFrame()

        if len(tickers) == 1:
            ticker = tickers[0]
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    prices = pd.DataFrame(data[ticker]['Close'])
                except Exception:
                    return pd.DataFrame()
            elif 'Close' in data.columns:
                prices = pd.DataFrame(data['Close'])
            else:
                return pd.DataFrame()

            prices.columns = [ticker]
            return prices

        prices = pd.DataFrame(index=data.index)
        if not isinstance(data.columns, pd.MultiIndex):
            return prices

        for ticker in tickers:
            try:
                prices[ticker] = data[ticker]['Close']
            except Exception:
                continue

        return prices
        
    def fetch_data(self):
        """
        Fetch stock data from Yahoo Finance
        """
        try:
            requested_tickers = [ticker.strip().upper() for ticker in self.tickers if ticker.strip()]
            if not requested_tickers:
                raise Exception("Please provide at least one ticker symbol.")

            # Download first and infer validity from real price history rather than .info,
            # which is often unavailable for otherwise valid Yahoo symbols.
            self.data = yf.download(
                requested_tickers,
                start=self.start_date,
                end=self.end_date,
                group_by='ticker',
                auto_adjust=True,
                progress=False,
                threads=False
            )
            
            if self.data.empty:
                raise Exception("No data downloaded. Please check date range and tickers.")

            prices = self._extract_close_prices(self.data, requested_tickers)
            prices = prices.dropna(axis=1, how='all')

            self.valid_tickers = list(prices.columns)
            invalid_tickers = [ticker for ticker in requested_tickers if ticker not in self.valid_tickers]

            if not self.valid_tickers:
                raise Exception(
                    f"No valid tickers found. Please check your symbols. Invalid tickers: {invalid_tickers}"
                )

            if invalid_tickers:
                warnings.warn(f"Invalid tickers ignored: {invalid_tickers}")

            if prices.empty:
                raise Exception("No price data available for the selected tickers")

            # Calculate returns
            self.returns = prices.pct_change().dropna()

            if self.returns.empty:
                raise Exception("Unable to calculate returns. Insufficient data.")

            # Get market caps
            self.market_caps = self._get_market_caps()

            return self.returns, self.market_caps

        except Exception as e:
            message = str(e)
            lowered = message.lower()
            if (
                "no timezone found" in lowered or
                "failed downloads" in lowered or
                "expecting value" in lowered or
                "json" in lowered
            ):
                raise Exception(
                    "Yahoo Finance request failed before price data was returned. "
                    "This is usually caused by an outdated yfinance version or a temporary Yahoo response issue. "
                    "Upgrade yfinance and try again."
                )
            raise Exception(f"Error fetching data: {message}")
    
    def _get_market_caps(self):
        """
        Get current market capitalizations
        """
        market_caps = {}
        for ticker in self.valid_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = {}

                try:
                    info = stock.fast_info or {}
                except Exception:
                    info = {}

                market_cap = info.get('market_cap')
                if market_cap is None:
                    try:
                        info = stock.info
                    except Exception:
                        info = {}
                    market_cap = info.get('marketCap')

                market_caps[ticker] = market_cap if market_cap is not None else 1e9
            except Exception:
                market_caps[ticker] = 1e9  # Default value if error
        
        return pd.Series(market_caps)
    
    def get_summary_stats(self):
        """
        Calculate summary statistics for returns
        """
        if self.returns is None:
            return None
        
        stats = pd.DataFrame({
            'Mean Return (%)': self.returns.mean() * 100,
            'Std Dev (%)': self.returns.std() * 100,
            'Sharpe Ratio': (self.returns.mean() / self.returns.std()) * np.sqrt(252),
            'Skewness': self.returns.skew(),
            'Kurtosis': self.returns.kurtosis()
        })
        
        return stats
    
    def get_correlation_matrix(self):
        """
        Calculate correlation matrix
        """
        if self.returns is None:
            return None
        
        return self.returns.corr()
    
    def get_covariance_matrix(self):
        """
        Calculate annualized covariance matrix
        """
        if self.returns is None:
            return None
        
        # Annualize by multiplying by 252 trading days
        return self.returns.cov() * 252
    
    @staticmethod
    def get_valid_tickers(tickers):
        """
        Check which tickers are valid
        """
        cleaned_tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
        if not cleaned_tickers:
            return []

        try:
            data = yf.download(
                cleaned_tickers,
                period="5d",
                group_by='ticker',
                auto_adjust=True,
                progress=False,
                threads=False
            )
        except Exception:
            return []

        prices = YahooFinanceData._extract_close_prices(data, cleaned_tickers)
        prices = prices.dropna(axis=1, how='all')
        return list(prices.columns)
    
    def get_price_history(self):
        """
        Get historical price data for visualization
        """
        if self.data is None:
            try:
                self.data = yf.download(
                    self.valid_tickers or self.tickers,
                    start=self.start_date,
                    end=self.end_date,
                    group_by='ticker',
                    auto_adjust=True,
                    progress=False,
                    threads=False
                )
            except Exception:
                return pd.DataFrame()
        
        try:
            tickers = self.valid_tickers or [ticker.strip().upper() for ticker in self.tickers if ticker.strip()]
            prices = self._extract_close_prices(self.data, tickers)
            if prices.empty:
                return pd.DataFrame()

            # Normalize prices to 100 for easier comparison
            normalized_prices = prices / prices.iloc[0] * 100
            return normalized_prices

        except Exception as e:
            print(f"Error getting price history: {e}")
            return pd.DataFrame()

    def get_ohlc_history(self):
        """
        Get OHLC data per ticker for candlestick and range-based charts.
        """
        tickers = self.valid_tickers or [ticker.strip().upper() for ticker in self.tickers if ticker.strip()]
        if not tickers:
            return {}

        if self.data is None:
            try:
                self.data = yf.download(
                    tickers,
                    start=self.start_date,
                    end=self.end_date,
                    group_by='ticker',
                    auto_adjust=False,
                    progress=False,
                    threads=False
                )
            except Exception:
                return {}

        try:
            ohlc_data = {}
            if len(tickers) == 1 and not isinstance(self.data.columns, pd.MultiIndex):
                available_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in self.data.columns]
                if available_cols:
                    ohlc_data[tickers[0]] = self.data[available_cols].dropna(how='all')
                return ohlc_data

            if not isinstance(self.data.columns, pd.MultiIndex):
                return {}

            for ticker in tickers:
                try:
                    ticker_df = self.data[ticker]
                    available_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in ticker_df.columns]
                    if available_cols:
                        ohlc_data[ticker] = ticker_df[available_cols].dropna(how='all')
                except Exception:
                    continue

            return ohlc_data
        except Exception as e:
            print(f"Error getting OHLC history: {e}")
            return {}
