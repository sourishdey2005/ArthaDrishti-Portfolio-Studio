"""
Portfolio Optimization Utilities
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Portfolio optimization with various constraints
    """
    
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.02):
        """
        Initialize portfolio optimizer
        
        Parameters:
        -----------
        expected_returns : np.array
            Expected returns for each asset
        cov_matrix : np.array
            Covariance matrix of returns
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculation
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)

    def _run_optimization(self, objective, bounds, constraints, initial_weights=None):
        """
        Run SLSQP optimization with consistent error handling.
        """
        if initial_weights is None:
            initial_weights = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            raise ValueError(result.message)

        return result.x
        
    def maximize_sharpe_ratio(self, allow_short=False, max_weight=0.3):
        """
        Maximize Sharpe ratio
        """
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if not allow_short:
            constraints.append({'type': 'ineq', 'fun': lambda x: x})
        
        bounds = [(0, max_weight) if not allow_short else (-max_weight, max_weight) 
                  for _ in range(self.n_assets)]
        
        def negative_sharpe(weights):
            portfolio_return = np.sum(weights * self.expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            portfolio_vol = np.sqrt(max(portfolio_variance, 0))
            if portfolio_vol <= 1e-12:
                return np.inf
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe
        
        initial_weights = np.ones(self.n_assets) / self.n_assets
        return self._run_optimization(negative_sharpe, bounds, constraints, initial_weights)
    
    def minimize_volatility(self, allow_short=False, max_weight=0.3):
        """
        Minimize portfolio volatility
        """
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if not allow_short:
            constraints.append({'type': 'ineq', 'fun': lambda x: x})
        
        bounds = [(0, max_weight) if not allow_short else (-max_weight, max_weight) 
                  for _ in range(self.n_assets)]
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        initial_weights = np.ones(self.n_assets) / self.n_assets
        return self._run_optimization(portfolio_variance, bounds, constraints, initial_weights)
    
    def optimize_with_target_return(self, target_return, allow_short=False, max_weight=0.3):
        """
        Optimize portfolio for a target return
        """
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(x * self.expected_returns) - target_return}
        ]
        
        if not allow_short:
            constraints.append({'type': 'ineq', 'fun': lambda x: x})
        
        bounds = [(0, max_weight) if not allow_short else (-max_weight, max_weight) 
                  for _ in range(self.n_assets)]
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        initial_weights = np.ones(self.n_assets) / self.n_assets
        return self._run_optimization(portfolio_variance, bounds, constraints, initial_weights)
    
    def calculate_portfolio_stats(self, weights):
        """
        Calculate portfolio statistics
        """
        portfolio_return = np.sum(weights * self.expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_volatility = np.sqrt(max(portfolio_variance, 0))
        sharpe_ratio = (
            (portfolio_return - self.risk_free_rate) / portfolio_volatility
            if portfolio_volatility > 1e-12 else np.nan
        )
        weighted_vols = np.sum(weights * np.sqrt(np.clip(np.diag(self.cov_matrix), a_min=0, a_max=None)))
        diversification_ratio = (
            weighted_vols / portfolio_volatility
            if portfolio_volatility > 1e-12 else np.nan
        )
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'variance': portfolio_variance,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': diversification_ratio
        }
    
    def get_efficient_frontier(self, n_points=50, allow_short=False, max_weight=0.3):
        """
        Calculate efficient frontier points
        """
        # Get min and max returns
        min_return = np.min(self.expected_returns)
        max_return = np.max(self.expected_returns)
        
        target_returns = np.linspace(min_return, max_return, n_points)
        frontier_volatilities = []
        frontier_weights = []
        
        for target in target_returns:
            try:
                weights = self.optimize_with_target_return(target, allow_short, max_weight)
                stats = self.calculate_portfolio_stats(weights)
                frontier_volatilities.append(stats['volatility'])
                frontier_weights.append(weights)
            except:
                frontier_volatilities.append(np.nan)
                frontier_weights.append(None)
        
        return target_returns, np.array(frontier_volatilities), frontier_weights
