"""
Black-Litterman Model Implementation
"""
import numpy as np
import pandas as pd
from scipy.linalg import inv


class BlackLittermanModel:
    """
    Black-Litterman Model for portfolio optimization with investor views
    """
    
    def __init__(self, returns, market_caps, delta=2.5, tau=0.025):
        """
        Initialize Black-Litterman Model
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns of assets
        market_caps : pd.Series
            Market capitalization weights
        delta : float
            Risk aversion coefficient
        tau : float
            Uncertainty in the prior
        """
        self.returns = returns
        self.n_assets = len(returns.columns)
        self.assets = returns.columns.tolist()
        
        # Market equilibrium
        self.market_weights = market_caps / market_caps.sum()
        self.cov_matrix = returns.cov().values
        
        # Risk aversion and uncertainty
        self.delta = delta
        self.tau = tau
        
        # Calculate implied excess returns (Pi)
        self.pi = self._calculate_implied_returns()
        
        # Initialize views
        self.views = None
        self.views_uncertainty = None
        self.P = None  # Pick matrix
        self.Q = None  # View vector
        
    def _calculate_implied_returns(self):
        """
        Calculate implied excess returns from market equilibrium
        """
        # Calculate portfolio variance
        portfolio_variance = np.dot(self.market_weights, 
                                   np.dot(self.cov_matrix, self.market_weights))
        
        # Implied returns using CAPM relationship
        implied_returns = self.delta * np.dot(self.cov_matrix, self.market_weights)
        
        return implied_returns
    
    def add_absolute_view(self, asset, expected_return, confidence=0.5):
        """
        Add absolute view on a single asset
        
        Parameters:
        -----------
        asset : str
            Asset symbol
        expected_return : float
            Expected return of the asset
        confidence : float
            Confidence level in the view (0-1)
        """
        if asset not in self.assets:
            raise ValueError(f"Asset {asset} not found in the universe")
        
        idx = self.assets.index(asset)
        
        if self.P is None:
            self.P = np.zeros((1, self.n_assets))
            self.Q = np.zeros(1)
            self.views_uncertainty = np.zeros((1, 1))
        else:
            self.P = np.vstack([self.P, np.zeros((1, self.n_assets))])
            self.Q = np.append(self.Q, 0)
            self.views_uncertainty = np.pad(self.views_uncertainty, 
                                           ((0, 1), (0, 1)), 
                                           mode='constant')
        
        self.P[-1, idx] = 1
        self.Q[-1] = expected_return
        
        # Convert confidence to uncertainty (higher confidence = lower uncertainty)
        uncertainty = (1 - confidence) * self.tau * np.diag(self.cov_matrix)[idx]
        self.views_uncertainty[-1, -1] = uncertainty
        
    def add_relative_view(self, asset_outperform, asset_underperform, 
                          expected_outperformance, confidence=0.5):
        """
        Add relative view between two assets
        
        Parameters:
        -----------
        asset_outperform : str
            Asset expected to outperform
        asset_underperform : str
            Asset expected to underperform
        expected_outperformance : float
            Expected outperformance of asset_outperform over asset_underperform
        confidence : float
            Confidence level in the view (0-1)
        """
        if asset_outperform not in self.assets or asset_underperform not in self.assets:
            raise ValueError("One or both assets not found in the universe")
        
        idx_out = self.assets.index(asset_outperform)
        idx_under = self.assets.index(asset_underperform)
        
        if self.P is None:
            self.P = np.zeros((1, self.n_assets))
            self.Q = np.zeros(1)
            self.views_uncertainty = np.zeros((1, 1))
        else:
            self.P = np.vstack([self.P, np.zeros((1, self.n_assets))])
            self.Q = np.append(self.Q, 0)
            self.views_uncertainty = np.pad(self.views_uncertainty, 
                                           ((0, 1), (0, 1)), 
                                           mode='constant')
        
        self.P[-1, idx_out] = 1
        self.P[-1, idx_under] = -1
        self.Q[-1] = expected_outperformance
        
        # Calculate uncertainty based on covariance
        variance = (self.cov_matrix[idx_out, idx_out] + 
                   self.cov_matrix[idx_under, idx_under] - 
                   2 * self.cov_matrix[idx_out, idx_under])
        uncertainty = (1 - confidence) * self.tau * variance
        self.views_uncertainty[-1, -1] = uncertainty
    
    def calculate_posterior(self):
        """
        Calculate posterior returns and covariance matrix using Black-Litterman formula
        """
        if self.P is None:
            # No views, return prior
            return self.pi, self.cov_matrix
        
        # Black-Litterman formula
        # μ_bl = [(τΣ)^-1 + P'Ω^-1 P]^-1 * [(τΣ)^-1 Π + P'Ω^-1 Q]
        # Σ_bl = Σ + [(τΣ)^-1 + P'Ω^-1 P]^-1
        
        tau_cov = self.tau * self.cov_matrix
        tau_cov_inv = inv(tau_cov)
        
        omega_inv = inv(self.views_uncertainty)
        
        # Calculate posterior mean
        A = tau_cov_inv + np.dot(self.P.T, np.dot(omega_inv, self.P))
        A_inv = inv(A)
        
        B = np.dot(tau_cov_inv, self.pi.reshape(-1, 1)) + \
            np.dot(self.P.T, np.dot(omega_inv, self.Q.reshape(-1, 1)))
        
        posterior_returns = np.dot(A_inv, B).flatten()
        
        # Calculate posterior covariance
        posterior_cov = self.cov_matrix + A_inv
        
        return posterior_returns, posterior_cov
    
    def get_optimal_weights(self, allow_short=False, max_weight=0.3):
        """
        Calculate optimal weights using mean-variance optimization
        """
        posterior_returns, posterior_cov = self.calculate_posterior()
        
        # Mean-variance optimization with constraints
        from scipy.optimize import minimize
        
        n = self.n_assets
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if not allow_short:
            constraints.append({'type': 'ineq', 'fun': lambda x: x})
        
        bounds = [(0, max_weight) if not allow_short else (-max_weight, max_weight) 
                  for _ in range(n)]
        
        def objective(weights):
            portfolio_return = np.sum(weights * posterior_returns)
            portfolio_variance = np.dot(weights.T, np.dot(posterior_cov, weights))
            return - (portfolio_return - 0.5 * self.delta * portfolio_variance)
        
        initial_weights = np.ones(n) / n
        result = minimize(objective, initial_weights, 
                         method='SLSQP', 
                         bounds=bounds, 
                         constraints=constraints)

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        return result.x
    
    def get_results_dataframe(self):
        """
        Get comprehensive results as DataFrame
        """
        prior_returns = self.pi
        posterior_returns, posterior_cov = self.calculate_posterior()
        
        # Calculate volatilities
        prior_vol = np.sqrt(np.diag(self.cov_matrix))
        posterior_vol = np.sqrt(np.diag(posterior_cov))
        
        # Get optimal weights
        optimal_weights = self.get_optimal_weights()
        
        results_df = pd.DataFrame({
            'Asset': self.assets,
            'Market Weight': self.market_weights * 100,
            'Prior Return (%)': prior_returns * 100,
            'Posterior Return (%)': posterior_returns * 100,
            'Prior Volatility (%)': prior_vol * 100,
            'Posterior Volatility (%)': posterior_vol * 100,
            'Optimal Weight (%)': optimal_weights * 100
        })
        
        return results_df
