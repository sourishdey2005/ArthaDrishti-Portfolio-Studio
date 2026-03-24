"""
Visualization utilities for the Black-Litterman Model
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None


class Visualizer:
    """
    Create interactive visualizations for portfolio analysis.
    """

    def __init__(self, theme_colors=None):
        self.theme_colors = theme_colors or {
            'primary': '#60a5fa',
            'secondary': '#f59e0b',
            'success': '#22c55e',
            'danger': '#f87171',
            'background': '#0f172a',
            'surface': '#111827',
            'grid': 'rgba(148, 163, 184, 0.18)',
            'text': '#f8fafc',
            'muted_text': '#cbd5e1'
        }

    def _update_layout(self, fig, title, xaxis_title='', yaxis_title=''):
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            template='plotly_dark',
            paper_bgcolor=self.theme_colors['background'],
            plot_bgcolor=self.theme_colors['surface'],
            font=dict(color=self.theme_colors['text']),
            title_font=dict(color=self.theme_colors['text']),
            hovermode='closest',
            margin=dict(l=40, r=20, t=60, b=40)
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor=self.theme_colors['grid'],
            zerolinecolor=self.theme_colors['grid'],
            color=self.theme_colors['muted_text']
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=self.theme_colors['grid'],
            zerolinecolor=self.theme_colors['grid'],
            color=self.theme_colors['muted_text']
        )
        return fig

    def _portfolio_returns(self, returns_df, weights=None):
        if weights is None:
            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
        return returns_df.mul(weights, axis=1).sum(axis=1)

    def _drawdown_series(self, returns_series):
        cumulative = (1 + returns_series).cumprod()
        rolling_max = cumulative.cummax()
        return cumulative / rolling_max - 1

    def _summary_stats(self, returns_df, risk_free_rate=0.02):
        annual_return = returns_df.mean() * 252
        annual_vol = returns_df.std() * np.sqrt(252)
        sharpe = (annual_return - risk_free_rate) / annual_vol.replace(0, np.nan)
        downside = returns_df.clip(upper=0).std() * np.sqrt(252)
        sortino = (annual_return - risk_free_rate) / downside.replace(0, np.nan)
        skewness = returns_df.skew()
        kurtosis = returns_df.kurtosis()
        var_95 = returns_df.apply(lambda col: np.percentile(col.dropna(), 5) if len(col.dropna()) else np.nan)
        cvar_95 = returns_df.apply(
            lambda col: col[col <= np.percentile(col.dropna(), 5)].mean() if len(col.dropna()) else np.nan
        )
        max_drawdown = returns_df.apply(
            lambda col: self._drawdown_series(col.fillna(0)).min() if len(col.dropna()) else np.nan
        )
        positive_days = (returns_df > 0).mean()
        negative_days = (returns_df < 0).mean()

        return pd.DataFrame({
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'VaR 95%': var_95,
            'CVaR 95%': cvar_95,
            'Max Drawdown': max_drawdown,
            'Positive Days': positive_days,
            'Negative Days': negative_days
        })

    def _bar_chart(self, series, title, yaxis_title, color=None):
        series = pd.Series(series).sort_values(ascending=False)
        fig = go.Figure(
            go.Bar(
                x=series.index,
                y=series.values,
                marker_color=color or self.theme_colors['primary'],
                text=np.round(series.values, 2),
                textposition='auto'
            )
        )
        return self._update_layout(fig, title, 'Asset', yaxis_title)

    def plot_returns_distribution(self, returns_df):
        fig = go.Figure()
        for column in returns_df.columns:
            fig.add_trace(go.Histogram(
                x=returns_df[column] * 100,
                name=column,
                opacity=0.65,
                nbinsx=50
            ))
        fig.update_layout(
            barmode='overlay',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return self._update_layout(fig, 'Returns Distribution', 'Daily Returns (%)', 'Frequency')

    def plot_correlation_heatmap(self, correlation_matrix):
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        fig.update_layout(width=800, height=600)
        return self._update_layout(fig, 'Asset Correlation Matrix')

    def plot_price_history(self, normalized_prices):
        fig = go.Figure()
        for column in normalized_prices.columns:
            fig.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[column],
                mode='lines',
                name=column,
                line=dict(width=2)
            ))
        fig.update_layout(hovermode='x unified', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return self._update_layout(fig, 'Normalized Price History (Base=100)', 'Date', 'Normalized Price')

    def plot_weights_comparison(self, market_weights, prior_weights, posterior_weights):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=market_weights.index, y=market_weights.values, name='Market Weights',
                             marker_color=self.theme_colors['primary']))
        fig.add_trace(go.Bar(x=prior_weights.index, y=prior_weights.values, name='Prior Weights',
                             marker_color=self.theme_colors['secondary']))
        fig.add_trace(go.Bar(x=posterior_weights.index, y=posterior_weights.values, name='Posterior Weights',
                             marker_color=self.theme_colors['success']))
        fig.update_layout(barmode='group', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return self._update_layout(fig, 'Portfolio Weights Comparison', 'Asset', 'Weight (%)')

    def plot_efficient_frontier(self, returns, volatilities, optimal_point=None):
        sharpe = np.divide(returns, volatilities, out=np.zeros_like(returns), where=volatilities > 0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=volatilities * 100,
            y=returns * 100,
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color=self.theme_colors['primary'], width=2),
            marker=dict(size=7, color=sharpe, colorscale='Viridis', showscale=True, colorbar=dict(title='Sharpe'))
        ))
        if optimal_point:
            fig.add_trace(go.Scatter(
                x=[optimal_point['volatility'] * 100],
                y=[optimal_point['return'] * 100],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(size=15, color=self.theme_colors['success'], symbol='star', line=dict(color='white', width=2))
            ))
        return self._update_layout(fig, 'Efficient Frontier', 'Annualized Volatility (%)', 'Annualized Return (%)')

    def plot_returns_comparison(self, prior_returns, posterior_returns, assets):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=assets, y=prior_returns * 100, name='Prior Returns',
                             marker_color=self.theme_colors['primary']))
        fig.add_trace(go.Bar(x=assets, y=posterior_returns * 100, name='Posterior Returns',
                             marker_color=self.theme_colors['secondary']))
        fig.update_layout(barmode='group')
        return self._update_layout(fig, 'Prior vs Posterior Expected Returns', 'Asset', 'Expected Return (%)')

    def plot_portfolio_allocation(self, weights, title='Portfolio Allocation'):
        weights_filtered = {k: v for k, v in weights.items() if v > 0.01}
        fig = go.Figure(data=[go.Pie(
            labels=list(weights_filtered.keys()),
            values=list(weights_filtered.values()),
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        return self._update_layout(fig, title)

    def plot_cumulative_returns(self, returns_df):
        cumulative = (1 + returns_df).cumprod() - 1
        fig = go.Figure()
        for column in cumulative.columns:
            fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative[column] * 100, mode='lines', name=column))
        fig.update_layout(hovermode='x unified')
        return self._update_layout(fig, 'Cumulative Returns', 'Date', 'Return (%)')

    def plot_drawdown_chart(self, returns_df):
        fig = go.Figure()
        for column in returns_df.columns:
            fig.add_trace(go.Scatter(
                x=returns_df.index,
                y=self._drawdown_series(returns_df[column].fillna(0)) * 100,
                mode='lines',
                name=column,
                fill='tozeroy'
            ))
        fig.update_layout(hovermode='x unified')
        return self._update_layout(fig, 'Drawdown Curves', 'Date', 'Drawdown (%)')

    def plot_rolling_volatility(self, returns_df, window=63):
        rolling_vol = returns_df.rolling(window).std() * np.sqrt(252) * 100
        fig = go.Figure()
        for column in rolling_vol.columns:
            fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[column], mode='lines', name=column))
        fig.update_layout(hovermode='x unified')
        return self._update_layout(fig, f'Rolling Volatility ({window}D)', 'Date', 'Volatility (%)')

    def plot_rolling_sharpe(self, returns_df, risk_free_rate=0.02, window=63):
        rolling_mean = returns_df.rolling(window).mean() * 252
        rolling_vol = returns_df.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_vol.replace(0, np.nan)
        fig = go.Figure()
        for column in rolling_sharpe.columns:
            fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe[column], mode='lines', name=column))
        fig.update_layout(hovermode='x unified')
        return self._update_layout(fig, f'Rolling Sharpe Ratio ({window}D)', 'Date', 'Sharpe Ratio')

    def plot_return_boxplot(self, returns_df):
        fig = go.Figure()
        for column in returns_df.columns:
            fig.add_trace(go.Box(y=returns_df[column] * 100, name=column, boxmean=True))
        return self._update_layout(fig, 'Return Dispersion by Asset', 'Asset', 'Daily Return (%)')

    def plot_return_violin(self, returns_df):
        fig = go.Figure()
        for column in returns_df.columns:
            fig.add_trace(go.Violin(y=returns_df[column] * 100, name=column, box_visible=True, meanline_visible=True))
        return self._update_layout(fig, 'Return Shape by Asset', 'Asset', 'Daily Return (%)')

    def plot_monthly_returns_heatmap(self, returns_df, asset=None):
        asset = asset or returns_df.columns[0]
        monthly = returns_df[asset].resample('M').apply(lambda x: (1 + x).prod() - 1)
        heatmap_df = monthly.to_frame('Return')
        heatmap_df['Year'] = heatmap_df.index.year
        heatmap_df['Month'] = heatmap_df.index.strftime('%b')
        pivot = heatmap_df.pivot(index='Year', columns='Month', values='Return')
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot = pivot.reindex(columns=month_order)
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values * 100,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot.values * 100, 2),
            texttemplate='%{text}'
        ))
        return self._update_layout(fig, f'Monthly Returns Heatmap: {asset}', 'Month', 'Year')

    def plot_annual_returns_bar(self, returns_df):
        annual = returns_df.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        fig = go.Figure()
        for column in annual.columns:
            fig.add_trace(go.Bar(x=annual.index.year, y=annual[column] * 100, name=column))
        fig.update_layout(barmode='group')
        return self._update_layout(fig, 'Annual Returns by Asset', 'Year', 'Return (%)')

    def plot_risk_return_scatter(self, returns_df, weights=None):
        stats = self._summary_stats(returns_df)
        size = np.full(len(stats), 20) if weights is None else np.clip(np.array(weights) * 120, 12, 40)
        fig = go.Figure(go.Scatter(
            x=stats['Annual Volatility'] * 100,
            y=stats['Annual Return'] * 100,
            mode='markers+text',
            text=stats.index,
            textposition='top center',
            marker=dict(size=size, color=stats['Sharpe Ratio'], colorscale='Viridis', showscale=True)
        ))
        return self._update_layout(fig, 'Risk vs Return Scatter', 'Annual Volatility (%)', 'Annual Return (%)')

    def plot_covariance_heatmap(self, cov_matrix, assets):
        cov_df = pd.DataFrame(cov_matrix, index=assets, columns=assets)
        fig = go.Figure(data=go.Heatmap(
            z=cov_df.values,
            x=cov_df.columns,
            y=cov_df.index,
            colorscale='Blues',
            text=np.round(cov_df.values, 4),
            texttemplate='%{text}'
        ))
        return self._update_layout(fig, 'Covariance Matrix', 'Asset', 'Asset')

    def plot_mean_returns_bar(self, returns_df):
        return self._bar_chart(returns_df.mean() * 252 * 100, 'Annualized Mean Returns', 'Return (%)')

    def plot_volatility_bar(self, returns_df):
        return self._bar_chart(returns_df.std() * np.sqrt(252) * 100, 'Annualized Volatility by Asset', 'Volatility (%)',
                               self.theme_colors['secondary'])

    def plot_sharpe_bar(self, returns_df, risk_free_rate=0.02):
        stats = self._summary_stats(returns_df, risk_free_rate)
        return self._bar_chart(stats['Sharpe Ratio'], 'Sharpe Ratio by Asset', 'Sharpe Ratio', self.theme_colors['success'])

    def plot_sortino_bar(self, returns_df, risk_free_rate=0.02):
        stats = self._summary_stats(returns_df, risk_free_rate)
        return self._bar_chart(stats['Sortino Ratio'], 'Sortino Ratio by Asset', 'Sortino Ratio', '#17a2b8')

    def plot_skewness_bar(self, returns_df):
        stats = self._summary_stats(returns_df)
        return self._bar_chart(stats['Skewness'], 'Skewness by Asset', 'Skewness', '#9467bd')

    def plot_kurtosis_bar(self, returns_df):
        stats = self._summary_stats(returns_df)
        return self._bar_chart(stats['Kurtosis'], 'Kurtosis by Asset', 'Kurtosis', '#8c564b')

    def plot_var_bar(self, returns_df):
        stats = self._summary_stats(returns_df)
        return self._bar_chart(stats['VaR 95%'] * 100, 'Value at Risk (95%)', 'VaR (%)', '#e74c3c')

    def plot_cvar_bar(self, returns_df):
        stats = self._summary_stats(returns_df)
        return self._bar_chart(stats['CVaR 95%'] * 100, 'Conditional VaR (95%)', 'CVaR (%)', '#c0392b')

    def plot_max_drawdown_bar(self, returns_df):
        stats = self._summary_stats(returns_df)
        return self._bar_chart(stats['Max Drawdown'] * 100, 'Maximum Drawdown by Asset', 'Max Drawdown (%)', '#34495e')

    def plot_positive_days_bar(self, returns_df):
        stats = self._summary_stats(returns_df)
        return self._bar_chart(stats['Positive Days'] * 100, 'Positive Return Days', 'Positive Days (%)', '#27ae60')

    def plot_negative_days_bar(self, returns_df):
        stats = self._summary_stats(returns_df)
        return self._bar_chart(stats['Negative Days'] * 100, 'Negative Return Days', 'Negative Days (%)', '#e67e22')

    def plot_best_worst_days(self, returns_df):
        best = returns_df.max() * 100
        worst = returns_df.min() * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=best.index, y=best.values, name='Best Day', marker_color=self.theme_colors['success']))
        fig.add_trace(go.Bar(x=worst.index, y=worst.values, name='Worst Day', marker_color=self.theme_colors['danger']))
        fig.update_layout(barmode='group')
        return self._update_layout(fig, 'Best and Worst Daily Moves', 'Asset', 'Return (%)')

    def plot_returns_scatter_matrix(self, returns_df):
        fig = px.scatter_matrix(
            returns_df.reset_index(drop=True),
            dimensions=returns_df.columns.tolist(),
            opacity=0.55
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        return self._update_layout(fig, 'Returns Scatter Matrix')

    def plot_rolling_correlation(self, returns_df, asset_x=None, asset_y=None, window=63):
        asset_x = asset_x or returns_df.columns[0]
        asset_y = asset_y or returns_df.columns[min(1, len(returns_df.columns) - 1)]
        rolling_corr = returns_df[asset_x].rolling(window).corr(returns_df[asset_y])
        fig = go.Figure(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode='lines', line=dict(color='#6f42c1')))
        return self._update_layout(fig, f'Rolling Correlation: {asset_x} vs {asset_y}', 'Date', 'Correlation')

    def plot_portfolio_growth(self, returns_df, weights=None, initial_value=10000):
        portfolio_returns = self._portfolio_returns(returns_df, weights)
        growth = initial_value * (1 + portfolio_returns).cumprod()
        fig = go.Figure(go.Scatter(x=growth.index, y=growth.values, mode='lines', fill='tozeroy'))
        return self._update_layout(fig, 'Portfolio Growth Simulation', 'Date', 'Portfolio Value')

    def plot_weight_treemap(self, weights):
        weights_series = pd.Series(weights)
        fig = px.treemap(
            names=weights_series.index,
            parents=['Portfolio'] * len(weights_series),
            values=weights_series.values,
            color=weights_series.values,
            color_continuous_scale='Blues'
        )
        return self._update_layout(fig, 'Allocation Treemap')

    def plot_weight_waterfall(self, weights):
        weights_series = pd.Series(weights).sort_values(ascending=False) * 100
        fig = go.Figure(go.Waterfall(
            x=weights_series.index,
            y=weights_series.values,
            measure=['relative'] * len(weights_series)
        ))
        return self._update_layout(fig, 'Allocation Waterfall', 'Asset', 'Weight (%)')

    def plot_risk_contribution_bar(self, weights, cov_matrix, assets):
        weights_arr = np.array(weights)
        portfolio_variance = np.dot(weights_arr.T, np.dot(cov_matrix, weights_arr))
        if portfolio_variance <= 0:
            contributions = np.zeros(len(weights_arr))
        else:
            contributions = weights_arr * (np.dot(cov_matrix, weights_arr) / portfolio_variance) * 100
        series = pd.Series(contributions, index=assets)
        return self._bar_chart(series, 'Risk Contribution by Asset', 'Risk Contribution (%)', self.theme_colors['danger'])

    def plot_risk_contribution_treemap(self, weights, cov_matrix, assets):
        weights_arr = np.array(weights)
        portfolio_variance = np.dot(weights_arr.T, np.dot(cov_matrix, weights_arr))
        if portfolio_variance <= 0:
            contributions = np.zeros(len(weights_arr))
        else:
            contributions = np.abs(weights_arr * (np.dot(cov_matrix, weights_arr) / portfolio_variance))
        series = pd.Series(contributions, index=assets)
        fig = px.treemap(
            names=series.index,
            parents=['Portfolio Risk'] * len(series),
            values=series.values,
            color=series.values,
            color_continuous_scale='Reds'
        )
        return self._update_layout(fig, 'Risk Contribution Treemap')

    def plot_frontier_weights_heatmap(self, frontier_weights, assets):
        valid_weights = [weights for weights in frontier_weights if weights is not None]
        if not valid_weights:
            return go.Figure()
        weight_df = pd.DataFrame(valid_weights, columns=assets)
        fig = go.Figure(data=go.Heatmap(
            z=weight_df.T.values * 100,
            x=list(range(1, len(weight_df) + 1)),
            y=assets,
            colorscale='Viridis'
        ))
        return self._update_layout(fig, 'Efficient Frontier Weights Heatmap', 'Frontier Point', 'Asset')

    def plot_weight_vs_return_scatter(self, weights, posterior_returns, assets):
        fig = go.Figure(go.Scatter(
            x=np.array(weights) * 100,
            y=np.array(posterior_returns) * 100,
            mode='markers+text',
            text=assets,
            textposition='top center',
            marker=dict(size=16, color=np.array(posterior_returns) * 100, colorscale='Tealgrn')
        ))
        return self._update_layout(fig, 'Weight vs Expected Return', 'Weight (%)', 'Expected Return (%)')

    def plot_cumulative_relative_performance(self, returns_df, weights=None):
        portfolio_returns = self._portfolio_returns(returns_df, weights)
        equal_weight_returns = self._portfolio_returns(returns_df)
        relative = ((1 + portfolio_returns).cumprod() / (1 + equal_weight_returns).cumprod() - 1) * 100
        fig = go.Figure(go.Scatter(x=relative.index, y=relative.values, mode='lines', line=dict(color='#20c997')))
        return self._update_layout(fig, 'Relative Performance vs Equal Weight', 'Date', 'Outperformance (%)')

    def plot_rolling_beta(self, returns_df, asset=None, benchmark=None, window=63):
        asset = asset or returns_df.columns[0]
        benchmark_series = benchmark if benchmark is not None else returns_df.mean(axis=1)
        rolling_cov = returns_df[asset].rolling(window).cov(benchmark_series)
        rolling_var = benchmark_series.rolling(window).var()
        beta = rolling_cov / rolling_var.replace(0, np.nan)
        fig = go.Figure(go.Scatter(x=beta.index, y=beta.values, mode='lines', line=dict(color='#fd7e14')))
        return self._update_layout(fig, f'Rolling Beta: {asset}', 'Date', 'Beta')

    def plot_return_rank_bar(self, returns_df):
        annual_return = returns_df.mean() * 252 * 100
        ranked = annual_return.sort_values(ascending=False)
        return self._bar_chart(ranked, 'Return Ranking', 'Annual Return (%)', '#198754')

    def plot_volatility_rank_bar(self, returns_df):
        annual_vol = returns_df.std() * np.sqrt(252) * 100
        ranked = annual_vol.sort_values(ascending=False)
        return self._bar_chart(ranked, 'Volatility Ranking', 'Annual Volatility (%)', '#dc3545')

    def plot_sharpe_rank_bar(self, returns_df, risk_free_rate=0.02):
        stats = self._summary_stats(returns_df, risk_free_rate)
        ranked = stats['Sharpe Ratio'].sort_values(ascending=False)
        return self._bar_chart(ranked, 'Sharpe Ranking', 'Sharpe Ratio', '#0d6efd')

    def plot_return_contribution_bar(self, returns_df, weights, assets):
        contribution = returns_df.mean() * 252 * np.array(weights) * 100
        contribution.index = assets
        return self._bar_chart(contribution, 'Return Contribution by Asset', 'Contribution (%)', '#6610f2')

    def plot_diversification_benefit(self, returns_df, weights, cov_matrix, assets):
        standalone_vol = np.sqrt(np.clip(np.diag(cov_matrix), a_min=0, a_max=None)) * np.array(weights) * 100
        portfolio_vol = np.sqrt(max(np.dot(np.array(weights).T, np.dot(cov_matrix, np.array(weights))), 0)) * 100
        series = pd.Series(standalone_vol, index=assets)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=series.index, y=series.values, name='Weighted Standalone Vol', marker_color='#6c757d'))
        fig.add_trace(go.Scatter(
            x=series.index,
            y=[portfolio_vol] * len(series),
            mode='lines',
            name='Portfolio Volatility',
            line=dict(color='#111827', dash='dash')
        ))
        return self._update_layout(fig, 'Diversification Benefit View', 'Asset', 'Volatility (%)')

    def plot_tail_risk_bubble(self, returns_df):
        stats = self._summary_stats(returns_df)
        fig = go.Figure(go.Scatter(
            x=stats['VaR 95%'] * 100,
            y=stats['CVaR 95%'] * 100,
            mode='markers+text',
            text=stats.index,
            textposition='top center',
            marker=dict(
                size=np.clip(stats['Annual Volatility'] * 180, 12, 40),
                color=stats['Max Drawdown'] * 100,
                colorscale='Magma',
                showscale=True
            )
        ))
        return self._update_layout(fig, 'Tail Risk Bubble Chart', 'VaR 95% (%)', 'CVaR 95% (%)')

    def plot_calendar_return_bars(self, returns_df, asset=None):
        asset = asset or returns_df.columns[0]
        monthly = returns_df[asset].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        colors = np.where(monthly >= 0, self.theme_colors['success'], self.theme_colors['danger'])
        fig = go.Figure(go.Bar(x=monthly.index, y=monthly.values, marker_color=colors))
        return self._update_layout(fig, f'Calendar Returns: {asset}', 'Month', 'Return (%)')

    def plot_frontier_return_distribution(self, target_returns):
        fig = go.Figure(go.Histogram(x=np.array(target_returns) * 100, nbinsx=25, marker_color='#0dcaf0'))
        return self._update_layout(fig, 'Efficient Frontier Return Distribution', 'Target Return (%)', 'Count')

    def plot_candlestick_chart(self, ohlc_df, ticker):
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        fig = go.Figure(data=[go.Candlestick(
            x=ohlc_df.index,
            open=ohlc_df['Open'],
            high=ohlc_df['High'],
            low=ohlc_df['Low'],
            close=ohlc_df['Close'],
            name=ticker
        )])
        fig.update_layout(xaxis_rangeslider_visible=False)
        return self._update_layout(fig, f'Candlestick Chart: {ticker}', 'Date', 'Price')

    def plot_ohlc_chart(self, ohlc_df, ticker):
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        fig = go.Figure(data=[go.Ohlc(
            x=ohlc_df.index,
            open=ohlc_df['Open'],
            high=ohlc_df['High'],
            low=ohlc_df['Low'],
            close=ohlc_df['Close'],
            name=ticker
        )])
        fig.update_layout(xaxis_rangeslider_visible=False)
        return self._update_layout(fig, f'OHLC Chart: {ticker}', 'Date', 'Price')

    def plot_volume_bars(self, ohlc_df, ticker):
        """Plot volume bars with professional coloring - green for up days, red for down days."""
        if ohlc_df is None or ohlc_df.empty or 'Volume' not in ohlc_df.columns:
            return go.Figure()
        
        # Color based on price movement
        colors = np.where(ohlc_df['Close'] >= ohlc_df['Open'], 
                         '#22c55e',  # Green for up days
                         '#ef4444')  # Red for down days
        
        fig = go.Figure()
        
        # Add volume bars with gradient coloring
        fig.add_trace(go.Bar(
            x=ohlc_df.index,
            y=ohlc_df['Volume'],
            marker_color=colors,
            marker_line_color=colors,
            marker_line_width=0.5,
            name='Volume',
            hovertemplate='<b>Date</b>: %{x}<br><b>Volume</b>: %{y:,.0f}<br><b>Price Change</b>: %{customdata}%<extra></extra>',
            customdata=np.where(ohlc_df['Close'] >= ohlc_df['Open'], 
                              ['▲ Up' if c >= o else '▼ Down' for c, o in zip(ohlc_df['Close'], ohlc_df['Open'])],
                              ['▼ Down' for _ in ohlc_df.index])
        ))
        
        # Add moving average line for volume
        if len(ohlc_df) > 20:
            volume_ma = ohlc_df['Volume'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=ohlc_df.index,
                y=volume_ma,
                mode='lines',
                name='20-day MA',
                line=dict(color='#fbbf24', width=2),
                hovertemplate='<b>Volume MA</b>: %{y:,.0f}<extra></extra>'
            ))
        
        # Calculate average volume for reference line
        avg_volume = ohlc_df['Volume'].mean()
        
        fig.update_layout(
            title=dict(
                text=f'<b>Volume Profile: {ticker}</b>',
                font=dict(size=18, color='#f8fafc')
            ),
            xaxis_title='Date',
            yaxis_title='Volume',
            template='plotly_dark',
            paper_bgcolor='#0f172a',
            plot_bgcolor='#111827',
            font=dict(color='#f8fafc'),
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=60),
            height=450
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', zerolinecolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', zerolinecolor='rgba(148, 163, 184, 0.15)', color='#94a3b8', tickformat=',d')
        
        return fig

    def plot_high_low_band(self, ohlc_df, ticker):
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df['High'], mode='lines', name='High', line=dict(color='#198754')))
        fig.add_trace(go.Scatter(
            x=ohlc_df.index,
            y=ohlc_df['Low'],
            mode='lines',
            name='Low',
            line=dict(color='#dc3545'),
            fill='tonexty',
            fillcolor='rgba(220,53,69,0.12)'
        ))
        return self._update_layout(fig, f'High-Low Trading Band: {ticker}', 'Date', 'Price')

    def plot_3d_risk_return_weights(self, returns_df, weights=None, risk_free_rate=0.02):
        stats = self._summary_stats(returns_df, risk_free_rate)
        if weights is None:
            weights = np.ones(len(stats)) / len(stats)
        fig = go.Figure(data=[go.Scatter3d(
            x=stats['Annual Volatility'] * 100,
            y=stats['Annual Return'] * 100,
            z=np.array(weights) * 100,
            mode='markers+text',
            text=stats.index,
            marker=dict(
                size=8,
                color=stats['Sharpe Ratio'],
                colorscale='Viridis',
                showscale=True,
                opacity=0.9
            )
        )])
        fig.update_layout(
            title='3D Risk-Return-Weight Map',
            scene=dict(
                xaxis_title='Volatility (%)',
                yaxis_title='Return (%)',
                zaxis_title='Weight (%)'
            ),
            template='plotly_dark'
        )
        return fig

    def plot_3d_frontier(self, returns, volatilities, frontier_weights, assets):
        valid = [(ret, vol, w) for ret, vol, w in zip(returns, volatilities, frontier_weights) if w is not None and not np.isnan(vol)]
        if not valid:
            return go.Figure()
        rets = np.array([item[0] for item in valid]) * 100
        vols = np.array([item[1] for item in valid]) * 100
        divers = []
        hover = []
        for ret, vol, weights in valid:
            weights = np.array(weights)
            effective_n = 1 / np.sum(np.square(weights)) if np.sum(np.square(weights)) > 0 else np.nan
            divers.append(effective_n)
            hover.append('<br>'.join(f'{asset}: {weight*100:.1f}%' for asset, weight in zip(assets, weights)))
        fig = go.Figure(data=[go.Scatter3d(
            x=vols,
            y=rets,
            z=np.array(divers),
            mode='markers+lines',
            text=hover,
            hovertemplate='Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<br>Effective N: %{z:.2f}<br>%{text}<extra></extra>',
            marker=dict(size=6, color=rets, colorscale='Turbo', showscale=True)
        )])
        fig.update_layout(
            title='3D Efficient Frontier',
            scene=dict(
                xaxis_title='Volatility (%)',
                yaxis_title='Return (%)',
                zaxis_title='Effective Number of Bets'
            ),
            template='plotly_dark'
        )
        return fig

    def plot_3d_covariance_surface(self, cov_matrix, assets):
        cov_df = pd.DataFrame(cov_matrix, index=assets, columns=assets)
        fig = go.Figure(data=[go.Surface(
            z=cov_df.values,
            x=list(range(len(assets))),
            y=list(range(len(assets))),
            colorscale='Blues'
        )])
        fig.update_layout(
            title='3D Covariance Surface',
            scene=dict(
                xaxis=dict(title='Asset X', tickmode='array', tickvals=list(range(len(assets))), ticktext=assets),
                yaxis=dict(title='Asset Y', tickmode='array', tickvals=list(range(len(assets))), ticktext=assets),
                zaxis_title='Covariance'
            ),
            template='plotly_dark'
        )
        return fig

    def plot_radar_performance(self, returns_df, risk_free_rate=0.02):
        stats = self._summary_stats(returns_df, risk_free_rate).fillna(0)
        metrics = ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Positive Days']
        fig = go.Figure()
        for asset in stats.index:
            values = stats.loc[asset, metrics].tolist()
            values.append(values[0])
            fig.add_trace(go.Scatterpolar(r=values, theta=metrics + [metrics[0]], fill='toself', name=asset))
        fig.update_layout(title='Radar Performance Profile', template='plotly_dark', paper_bgcolor=self.theme_colors['background'], font=dict(color=self.theme_colors['text']))
        return fig

    def plot_parallel_coordinates(self, returns_df, risk_free_rate=0.02):
        stats = self._summary_stats(returns_df, risk_free_rate).reset_index().rename(columns={'index': 'Asset'})
        stats = stats.replace([np.inf, -np.inf], np.nan).fillna(0)
        fig = px.parallel_coordinates(
            stats,
            color='Sharpe Ratio',
            dimensions=['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown'],
            labels={'Annual Return': 'Ann Return', 'Annual Volatility': 'Ann Vol', 'Max Drawdown': 'Max DD'},
            color_continuous_scale=px.colors.diverging.Tealrose
        )
        fig.update_layout(title='Parallel Coordinates Risk Profile', template='plotly_dark', paper_bgcolor=self.theme_colors['background'], font=dict(color=self.theme_colors['text']))
        return fig

    def plot_sunburst_allocation(self, weights, posterior_returns, assets):
        weights_arr = np.array(weights)
        returns_arr = np.array(posterior_returns)
        risk_bucket = np.where(returns_arr >= np.median(returns_arr), 'Higher Return', 'Lower Return')
        allocation_bucket = np.where(weights_arr >= np.median(weights_arr), 'Core', 'Satellite')
        df = pd.DataFrame({
            'Bucket': allocation_bucket,
            'Return Bucket': risk_bucket,
            'Asset': assets,
            'Weight': np.maximum(weights_arr * 100, 0.001)
        })
        fig = px.sunburst(df, path=['Bucket', 'Return Bucket', 'Asset'], values='Weight', color='Weight', color_continuous_scale='Blues')
        fig.update_layout(title='Allocation Sunburst', template='plotly_dark', paper_bgcolor=self.theme_colors['background'], font=dict(color=self.theme_colors['text']))
        return fig

    def plot_sankey_allocation_flow(self, market_weights, optimal_weights, assets):
        market_weights = np.array(market_weights) * 100
        optimal_weights = np.array(optimal_weights) * 100
        labels = ['Market'] + [f'{asset} (M)' for asset in assets] + [f'{asset} (O)' for asset in assets] + ['Optimal']
        source = []
        target = []
        value = []
        for idx, asset in enumerate(assets):
            source.append(0)
            target.append(1 + idx)
            value.append(max(market_weights[idx], 0.001))
        optimal_root = len(labels) - 1
        offset = 1 + len(assets)
        for idx, asset in enumerate(assets):
            source.append(offset + idx)
            target.append(optimal_root)
            value.append(max(optimal_weights[idx], 0.001))
        for idx, asset in enumerate(assets):
            source.append(1 + idx)
            target.append(offset + idx)
            value.append(max(abs(optimal_weights[idx] - market_weights[idx]), 0.001))
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=labels, pad=14, thickness=14),
            link=dict(source=source, target=target, value=value)
        )])
        fig.update_layout(title='Market to Optimal Allocation Flow', template='plotly_dark', paper_bgcolor=self.theme_colors['background'], font=dict(color=self.theme_colors['text']))
        return fig

    def plot_return_heatmap_by_weekday(self, returns_df, asset=None):
        asset = asset or returns_df.columns[0]
        series = returns_df[asset].dropna()
        frame = pd.DataFrame({'Return': series})
        frame['Month'] = frame.index.strftime('%b')
        frame['Weekday'] = frame.index.strftime('%a')
        heat = frame.pivot_table(index='Weekday', columns='Month', values='Return', aggfunc='mean')
        weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        heat = heat.reindex(index=weekday_order, columns=month_order)
        fig = go.Figure(data=go.Heatmap(
            z=heat.values * 100,
            x=heat.columns,
            y=heat.index,
            colorscale='RdYlGn',
            zmid=0
        ))
        return self._update_layout(fig, f'Weekday Seasonality Heatmap: {asset}', 'Month', 'Weekday')

    def plot_candlestick_with_volume(self, ohlc_df, ticker):
        if ohlc_df is None or ohlc_df.empty or 'Volume' not in ohlc_df.columns:
            return go.Figure()
        colors = np.where(ohlc_df['Close'] >= ohlc_df['Open'], self.theme_colors['success'], self.theme_colors['danger'])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_trace(go.Candlestick(
            x=ohlc_df.index,
            open=ohlc_df['Open'],
            high=ohlc_df['High'],
            low=ohlc_df['Low'],
            close=ohlc_df['Close'],
            name=ticker
        ), row=1, col=1)
        fig.add_trace(go.Bar(x=ohlc_df.index, y=ohlc_df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', title=f'Candlestick + Volume: {ticker}', paper_bgcolor=self.theme_colors['background'], font=dict(color=self.theme_colors['text']))
        return fig

    def plot_daily_range_bar(self, ohlc_df, ticker):
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        daily_range = (ohlc_df['High'] - ohlc_df['Low']).tail(40)
        fig = go.Figure(go.Bar(x=daily_range.index, y=daily_range.values, marker_color='#f59e0b'))
        return self._update_layout(fig, f'Daily Trading Range: {ticker}', 'Date', 'Price Range')

    def plot_gap_bar(self, ohlc_df, ticker):
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        gaps = ((ohlc_df['Open'] - ohlc_df['Close'].shift(1)) / ohlc_df['Close'].shift(1) * 100).dropna().tail(40)
        colors = np.where(gaps >= 0, self.theme_colors['success'], self.theme_colors['danger'])
        fig = go.Figure(go.Bar(x=gaps.index, y=gaps.values, marker_color=colors))
        return self._update_layout(fig, f'Opening Gap Analysis: {ticker}', 'Date', 'Gap (%)')

    def plot_close_vs_volume_scatter(self, ohlc_df, ticker):
        if ohlc_df is None or ohlc_df.empty or 'Volume' not in ohlc_df.columns:
            return go.Figure()
        fig = go.Figure(go.Scatter(
            x=ohlc_df['Volume'],
            y=ohlc_df['Close'],
            mode='markers',
            marker=dict(
                size=np.clip((ohlc_df['High'] - ohlc_df['Low']).fillna(0) * 15, 8, 24),
                color=((ohlc_df['Close'] - ohlc_df['Open']) / ohlc_df['Open']).fillna(0) * 100,
                colorscale='RdYlGn',
                showscale=True
            ),
            text=ohlc_df.index.strftime('%Y-%m-%d')
        ))
        return self._update_layout(fig, f'Close vs Volume: {ticker}', 'Volume', 'Close Price')

    def plot_quarterly_returns_bar(self, returns_df):
        quarterly = returns_df.resample('Q').apply(lambda values: (1 + values).prod() - 1) * 100
        quarter_labels = quarterly.index.to_period('Q').astype(str)
        fig = go.Figure()
        for column in quarterly.columns:
            fig.add_trace(go.Bar(x=quarter_labels, y=quarterly[column], name=column))
        fig.update_layout(barmode='group')
        return self._update_layout(fig, 'Quarterly Returns', 'Quarter', 'Return (%)')

    def plot_monthly_boxplot(self, returns_df):
        monthly = returns_df.resample('M').apply(lambda values: (1 + values).prod() - 1) * 100
        fig = go.Figure()
        for column in monthly.columns:
            fig.add_trace(go.Box(y=monthly[column], name=column, boxmean=True))
        return self._update_layout(fig, 'Monthly Return Distribution', 'Asset', 'Monthly Return (%)')

    def plot_return_quantile_band(self, returns_df):
        quantiles = pd.DataFrame({
            '10th': returns_df.quantile(0.10) * 100,
            'Median': returns_df.quantile(0.50) * 100,
            '90th': returns_df.quantile(0.90) * 100
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(x=quantiles.index, y=quantiles['90th'] - quantiles['10th'], base=quantiles['10th'],
                             marker_color='rgba(31,119,180,0.25)', name='10th-90th Range'))
        fig.add_trace(go.Scatter(x=quantiles.index, y=quantiles['Median'], mode='markers', marker=dict(size=12, color='#1f77b4'), name='Median'))
        return self._update_layout(fig, 'Return Quantile Band', 'Asset', 'Daily Return (%)')

    def plot_expanding_sharpe(self, returns_df, risk_free_rate=0.02):
        fig = go.Figure()
        for column in returns_df.columns:
            expanding_mean = returns_df[column].expanding().mean() * 252
            expanding_vol = returns_df[column].expanding().std() * np.sqrt(252)
            expanding_sharpe = (expanding_mean - risk_free_rate) / expanding_vol.replace(0, np.nan)
            fig.add_trace(go.Scatter(x=returns_df.index, y=expanding_sharpe, mode='lines', name=column))
        return self._update_layout(fig, 'Expanding Sharpe Ratio', 'Date', 'Sharpe Ratio')

    def plot_rolling_skewness(self, returns_df, window=63):
        fig = go.Figure()
        for column in returns_df.columns:
            fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df[column].rolling(window).skew(), mode='lines', name=column))
        return self._update_layout(fig, f'Rolling Skewness ({window}D)', 'Date', 'Skewness')

    def plot_rolling_kurtosis(self, returns_df, window=63):
        fig = go.Figure()
        for column in returns_df.columns:
            fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df[column].rolling(window).kurt(), mode='lines', name=column))
        return self._update_layout(fig, f'Rolling Kurtosis ({window}D)', 'Date', 'Kurtosis')

    def plot_return_autocorrelation(self, returns_df, max_lag=10):
        lags = list(range(1, max_lag + 1))
        fig = go.Figure()
        for column in returns_df.columns:
            acf_values = [returns_df[column].autocorr(lag=lag) for lag in lags]
            fig.add_trace(go.Bar(x=lags, y=acf_values, name=column))
        fig.update_layout(barmode='group')
        return self._update_layout(fig, 'Return Autocorrelation', 'Lag', 'Autocorrelation')

    def plot_lag_scatter(self, returns_df, asset=None, lag=1):
        asset = asset or returns_df.columns[0]
        frame = pd.DataFrame({
            'Lagged': returns_df[asset].shift(lag) * 100,
            'Current': returns_df[asset] * 100
        }).dropna()
        fig = go.Figure(go.Scatter(
            x=frame['Lagged'],
            y=frame['Current'],
            mode='markers',
            marker=dict(size=9, color=frame['Current'], colorscale='RdBu')
        ))
        return self._update_layout(fig, f'Lag Scatter: {asset}', f'Return t-{lag} (%)', 'Return t (%)')

    def plot_drawdown_duration_bar(self, returns_df):
        durations = {}
        for column in returns_df.columns:
            drawdown = self._drawdown_series(returns_df[column].fillna(0))
            underwater = drawdown < 0
            longest = current = 0
            for value in underwater:
                current = current + 1 if value else 0
                longest = max(longest, current)
            durations[column] = longest
        return self._bar_chart(pd.Series(durations), 'Longest Drawdown Duration', 'Days', '#7c3aed')

    def plot_underwater_heatmap(self, returns_df):
        drawdowns = returns_df.apply(lambda column: self._drawdown_series(column.fillna(0)) * 100)
        fig = go.Figure(data=go.Heatmap(
            z=drawdowns.T.values,
            x=drawdowns.index,
            y=drawdowns.columns,
            colorscale='RdBu',
            zmax=0,
            zmin=np.nanmin(drawdowns.T.values) if len(drawdowns) else -10
        ))
        return self._update_layout(fig, 'Underwater Drawdown Heatmap', 'Date', 'Asset')

    def plot_correlation_bubble(self, correlation_matrix):
        corr = correlation_matrix.copy()
        fig = go.Figure()
        for i, row_name in enumerate(corr.index):
            for j, col_name in enumerate(corr.columns):
                fig.add_trace(go.Scatter(
                    x=[col_name],
                    y=[row_name],
                    mode='markers',
                    marker=dict(size=abs(corr.iloc[i, j]) * 45 + 8, color=corr.iloc[i, j], colorscale='RdBu', cmin=-1, cmax=1, showscale=(i == 0 and j == 0)),
                    text=[f'{corr.iloc[i, j]:.2f}'],
                    hovertemplate='%{y} vs %{x}<br>Corr: %{text}<extra></extra>',
                    showlegend=False
                ))
        return self._update_layout(fig, 'Correlation Bubble Matrix', 'Asset', 'Asset')

    def plot_beta_heatmap(self, returns_df):
        assets = returns_df.columns
        beta_matrix = pd.DataFrame(index=assets, columns=assets, dtype=float)
        for asset in assets:
            for benchmark in assets:
                variance = returns_df[benchmark].var()
                beta_matrix.loc[asset, benchmark] = returns_df[asset].cov(returns_df[benchmark]) / variance if variance else np.nan
        fig = go.Figure(data=go.Heatmap(z=beta_matrix.values, x=assets, y=assets, colorscale='Viridis'))
        return self._update_layout(fig, 'Pairwise Beta Heatmap', 'Benchmark Asset', 'Asset')

    def plot_weight_bubble_chart(self, weights, posterior_returns, cov_matrix, assets):
        vol = np.sqrt(np.clip(np.diag(cov_matrix), a_min=0, a_max=None)) * 100
        fig = go.Figure(go.Scatter(
            x=vol,
            y=np.array(posterior_returns) * 100,
            mode='markers+text',
            text=assets,
            textposition='top center',
            marker=dict(size=np.clip(np.array(weights) * 200, 12, 50), color=np.array(weights) * 100, colorscale='Blues', showscale=True)
        ))
        return self._update_layout(fig, 'Weight Bubble Map', 'Volatility (%)', 'Expected Return (%)')

    def plot_active_weight_bar(self, market_weights, optimal_weights, assets):
        active = (np.array(optimal_weights) - np.array(market_weights)) * 100
        colors = np.where(active >= 0, self.theme_colors['success'], self.theme_colors['danger'])
        fig = go.Figure(go.Bar(x=assets, y=active, marker_color=colors))
        return self._update_layout(fig, 'Active Weights vs Market', 'Asset', 'Active Weight (%)')

    def plot_concentration_polar(self, weights, assets):
        fig = go.Figure(go.Barpolar(
            r=np.array(weights) * 100,
            theta=assets,
            marker=dict(color=np.array(weights) * 100, colorscale='Plasma', showscale=True)
        ))
        fig.update_layout(title='Allocation Polar View', template='plotly_dark', paper_bgcolor=self.theme_colors['background'], font=dict(color=self.theme_colors['text']))
        return fig

    def plot_frontier_sharpe_area(self, returns, volatilities, risk_free_rate=0.02):
        sharpe = np.divide(returns - risk_free_rate, volatilities, out=np.zeros_like(returns), where=volatilities > 0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=volatilities * 100, y=sharpe, mode='lines', fill='tozeroy', line=dict(color='#0f766e')))
        return self._update_layout(fig, 'Frontier Sharpe Profile', 'Volatility (%)', 'Sharpe Ratio')

    def plot_return_polar_bar(self, returns_df):
        annual_returns = returns_df.mean() * 252 * 100
        fig = go.Figure(go.Barpolar(
            r=annual_returns.values,
            theta=annual_returns.index,
            marker=dict(color=annual_returns.values, colorscale='Turbo', showscale=True)
        ))
        fig.update_layout(title='Return Polar Chart', template='plotly_dark', paper_bgcolor=self.theme_colors['background'], font=dict(color=self.theme_colors['text']))
        return fig

    def plot_regime_strip(self, returns_df, asset=None, window=21):
        asset = asset or returns_df.columns[0]
        rolling_mean = returns_df[asset].rolling(window).mean() * 252 * 100
        regime = pd.cut(rolling_mean, bins=[-np.inf, -5, 5, np.inf], labels=['Weak', 'Neutral', 'Strong'])
        mapping = {'Weak': -1, 'Neutral': 0, 'Strong': 1}
        regime_values = regime.map(mapping).fillna(0)
        fig = go.Figure(go.Heatmap(
            z=[regime_values.values],
            x=returns_df.index,
            y=[asset],
            colorscale=[[0, '#dc2626'], [0.5, '#facc15'], [1, '#16a34a']],
            showscale=False
        ))
        return self._update_layout(fig, f'Return Regime Strip: {asset}', 'Date', '')

    @staticmethod
    def plot_matplotlib_correlation(correlation_matrix):
        if sns is None:
            raise ImportError("seaborn is required for the matplotlib correlation fallback")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', center=0, fmt='.2f', square=True, ax=ax)
        ax.set_title('Asset Correlation Matrix')
        plt.tight_layout()
        return fig

    # =========================================
    # NEW UNIQUE VISUALIZATIONS (30 new charts)
    # =========================================

    def plot_rsi_indicator(self, ohlc_df, ticker, period=14):
        """Relative Strength Index (RSI) - Momentum oscillator showing overbought/oversold conditions."""
        if ohlc_df is None or ohlc_df.empty or 'Close' not in ohlc_df.columns:
            return go.Figure()
        
        delta = ohlc_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=rsi, mode='lines', name='RSI', line=dict(color='#8b5cf6', width=2)))
        fig.add_hline(y=70, line_dash='dash', line_color='#ef4444', annotation_text='Overbought')
        fig.add_hline(y=30, line_dash='dash', line_color='#22c55e', annotation_text='Oversold')
        fig.add_hline(y=50, line_dash='dot', line_color='#64748b')
        
        fig.update_layout(
            title=dict(text=f'<b>RSI ({period}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='RSI', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            yaxis=dict(range=[0, 100]), height=350,
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_macd_indicator(self, ohlc_df, ticker, fast=12, slow=26, signal=9):
        """MACD (Moving Average Convergence Divergence) - Trend following momentum indicator."""
        if ohlc_df is None or ohlc_df.empty or 'Close' not in ohlc_df.columns:
            return go.Figure()
        
        ema_fast = ohlc_df['Close'].ewm(span=fast).mean()
        ema_slow = ohlc_df['Close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        colors = np.where(histogram >= 0, '#22c55e', '#ef4444')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ohlc_df.index, y=histogram, marker_color=colors, name='Histogram'))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=macd_line, mode='lines', name='MACD', line=dict(color='#3b82f6', width=2)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=signal_line, mode='lines', name='Signal', line=dict(color='#f59e0b', width=2)))
        
        fig.update_layout(
            title=dict(text=f'<b>MACD ({fast},{slow},{signal}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Value', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=350, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_bollinger_bands(self, ohlc_df, ticker, window=20, num_std=2):
        """Bollinger Bands - Volatility bands around a moving average."""
        if ohlc_df is None or ohlc_df.empty or 'Close' not in ohlc_df.columns:
            return go.Figure()
        
        ma = ohlc_df['Close'].rolling(window=window).mean()
        std = ohlc_df['Close'].rolling(window=window).std()
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df['Close'], mode='lines', name='Close', line=dict(color='#f8fafc', width=1.5)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=upper_band, mode='lines', name='Upper Band', line=dict(color='#ef4444', width=1)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ma, mode='lines', name='MA', line=dict(color='#fbbf24', width=1.5)))
        fig.add_trace(go.Scatter(
            x=ohlc_df.index, 
            y=lower_band, 
            mode='lines', 
            name='Lower Band', 
            line=dict(color='#22c55e', width=1), 
            fill='tonexty', 
            fillcolor="rgba(34, 197, 94, 0.1)"
        ))
        
        fig.update_layout(
            title=dict(text=f'<b>Bollinger Bands ({window},{num_std}σ): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Price', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=400, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_stochastic_oscillator(self, ohlc_df, ticker, k_period=14, d_period=3):
        """Stochastic Oscillator - Momentum indicator comparing closing price to price range."""
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        
        low_min = ohlc_df['Low'].rolling(window=k_period).min()
        high_max = ohlc_df['High'].rolling(window=k_period).max()
        k_percent = 100 * (ohlc_df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=k_percent, mode='lines', name='%K', line=dict(color='#3b82f6', width=2)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=d_percent, mode='lines', name='%D', line=dict(color='#f59e0b', width=2)))
        fig.add_hline(y=80, line_dash='dash', line_color='#ef4444', annotation_text='Overbought')
        fig.add_hline(y=20, line_dash='dash', line_color='#22c55e', annotation_text='Oversold')
        
        fig.update_layout(
            title=dict(text=f'<b>Stochastic Oscillator ({k_period},{d_period}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Value', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            yaxis=dict(range=[0, 100]), height=350,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_on_balance_volume(self, ohlc_df, ticker):
        """On Balance Volume (OBV) - Cumulative volume indicator showing buying/selling pressure."""
        if ohlc_df is None or ohlc_df.empty or 'Volume' not in ohlc_df.columns:
            return go.Figure()
        
        obv = pd.Series(index=ohlc_df.index, dtype=float)
        obv.iloc[0] = ohlc_df['Volume'].iloc[0]
        for i in range(1, len(ohlc_df)):
            if ohlc_df['Close'].iloc[i] > ohlc_df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + ohlc_df['Volume'].iloc[i]
            elif ohlc_df['Close'].iloc[i] < ohlc_df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - ohlc_df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        colors = np.where(ohlc_df['Close'].diff() >= 0, '#22c55e', '#ef4444')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ohlc_df.index, y=ohlc_df['Volume'], marker_color=colors, name='Volume', opacity=0.5))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=obv, mode='lines', name='OBV', line=dict(color='#8b5cf6', width=2.5)))
        
        fig.update_layout(
            title=dict(text=f'<b>On Balance Volume: {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='OBV / Volume', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=400, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8', tickformat=',d')
        return fig

    def plot_average_true_range(self, ohlc_df, ticker, period=14):
        """Average True Range (ATR) - Volatility indicator measuring price movement."""
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        
        high_low = ohlc_df['High'] - ohlc_df['Low']
        high_close = np.abs(ohlc_df['High'] - ohlc_df['Close'].shift())
        low_close = np.abs(ohlc_df['Low'] - ohlc_df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=atr, mode='lines', name='ATR', fill='tozeroy', line=dict(color='#f97316', width=2)))
        
        fig.update_layout(
            title=dict(text=f'<b>Average True Range ({period}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='ATR', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=350, margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_money_flow_index(self, ohlc_df, ticker, period=14):
        """Money Flow Index (MFI) - Volume-weighted RSI showing money flow in/out."""
        if ohlc_df is None or ohlc_df.empty or 'Volume' not in ohlc_df.columns:
            return go.Figure()
        
        typical_price = (ohlc_df['High'] + ohlc_df['Low'] + ohlc_df['Close']) / 3
        money_flow = typical_price * ohlc_df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, np.nan)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=mfi, mode='lines', name='MFI', line=dict(color='#06b6d4', width=2)))
        fig.add_hline(y=80, line_dash='dash', line_color='#ef4444', annotation_text='Overbought')
        fig.add_hline(y=20, line_dash='dash', line_color='#22c55e', annotation_text='Oversold')
        
        fig.update_layout(
            title=dict(text=f'<b>Money Flow Index ({period}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='MFI', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            yaxis=dict(range=[0, 100]), height=350,
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_williams_r(self, ohlc_df, ticker, period=14):
        """Williams %R - Momentum indicator measuring overbought/oversold levels."""
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        
        highest_high = ohlc_df['High'].rolling(window=period).max()
        lowest_low = ohlc_df['Low'].rolling(window=period).min()
        williams_r = -100 * (highest_high - ohlc_df['Close']) / (highest_high - lowest_low).replace(0, np.nan)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=williams_r, mode='lines', name='Williams %R', line=dict(color='#ec4899', width=2)))
        fig.add_hline(y=-20, line_dash='dash', line_color='#ef4444', annotation_text='Overbought')
        fig.add_hline(y=-80, line_dash='dash', line_color='#22c55e', annotation_text='Oversold')
        
        fig.update_layout(
            title=dict(text=f'<b>Williams %R ({period}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Williams %R', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            yaxis=dict(range=[-100, 0]), height=350,
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_commodity_channel_index(self, ohlc_df, ticker, period=20):
        """Commodity Channel Index (CCI) - Identifies cyclical trends."""
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        
        typical_price = (ohlc_df['High'] + ohlc_df['Low'] + ohlc_df['Close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=cci, mode='lines', name='CCI', line=dict(color='#84cc16', width=2)))
        fig.add_hline(y=100, line_dash='dash', line_color='#ef4444')
        fig.add_hline(y=-100, line_dash='dash', line_color='#22c55e')
        fig.add_hline(y=0, line_dash='dot', line_color='#64748b')
        
        fig.update_layout(
            title=dict(text=f'<b>Commodity Channel Index ({period}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='CCI', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=350, margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_price_rate_of_change(self, ohlc_df, ticker, period=12):
        """Price Rate of Change (ROC) - Momentum oscillator showing percentage change."""
        if ohlc_df is None or ohlc_df.empty or 'Close' not in ohlc_df.columns:
            return go.Figure()
        
        roc = ((ohlc_df['Close'] - ohlc_df['Close'].shift(period)) / ohlc_df['Close'].shift(period)) * 100
        
        colors = np.where(roc >= 0, '#22c55e', '#ef4444')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ohlc_df.index, y=roc, marker_color=colors, name='ROC'))
        fig.add_hline(y=0, line_dash='dot', line_color='#64748b')
        
        fig.update_layout(
            title=dict(text=f'<b>Price Rate of Change ({period}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='ROC (%)', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=350, margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_trix_indicator(self, ohlc_df, ticker, period=15):
        """TRIX - Triple exponential average showing rate of change of triple smoothed price."""
        if ohlc_df is None or ohlc_df.empty or 'Close' not in ohlc_df.columns:
            return go.Figure()
        
        ema1 = ohlc_df['Close'].ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        trix = ema3.pct_change() * 100
        
        colors = np.where(trix >= 0, '#22c55e', '#ef4444')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ohlc_df.index, y=trix, marker_color=colors, name='TRIX'))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=trix.rolling(9).mean(), mode='lines', name='Signal', line=dict(color='#fbbf24', width=2)))
        fig.add_hline(y=0, line_dash='dot', line_color='#64748b')
        
        fig.update_layout(
            title=dict(text=f'<b>TRIX ({period}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='TRIX', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=350, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_donchian_channels(self, ohlc_df, ticker, upper_period=20, lower_period=20):
        """Donchian Channels - Price channel showing highest high and lowest low."""
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        
        upper = ohlc_df['High'].rolling(window=upper_period).max()
        lower = ohlc_df['Low'].rolling(window=lower_period).min()
        middle = (upper + lower) / 2
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df['Close'], mode='lines', name='Close', line=dict(color='#f8fafc', width=1.5)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=upper, mode='lines', name='Upper Channel', line=dict(color='#ef4444', width=1)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=middle, mode='lines', name='Middle', line=dict(color='#fbbf24', width=1.5)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=lower, mode='lines', name='Lower Channel', line=dict(color='#22c55e', width=1), fill='tonexty', fillcolor='rgba(34, 197, 94, 0.1)'))
        
        fig.update_layout(
            title=dict(text=f'<b>Donchian Channels ({upper_period},{lower_period}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Price', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=400, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_vwap_indicator(self, ohlc_df, ticker):
        """Volume Weighted Average Price (VWAP) - Average price weighted by volume."""
        if ohlc_df is None or ohlc_df.empty or 'Volume' not in ohlc_df.columns:
            return go.Figure()
        
        typical_price = (ohlc_df['High'] + ohlc_df['Low'] + ohlc_df['Close']) / 3
        vwap = (typical_price * ohlc_df['Volume']).cumsum() / ohlc_df['Volume'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=ohlc_df.index, open=ohlc_df['Open'], high=ohlc_df['High'], low=ohlc_df['Low'], close=ohlc_df['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=vwap, mode='lines', name='VWAP', line=dict(color='#fbbf24', width=3)))
        
        fig.update_layout(
            title=dict(text=f'<b>VWAP: {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Price', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=450, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_ichimoku_cloud(self, ohlc_df, ticker):
        """Ichimoku Cloud - Multi-component indicator showing support/resistance and trend."""
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        
        nine_period_high = ohlc_df['High'].rolling(window=9).max()
        nine_period_low = ohlc_df['Low'].rolling(window=9).min()
        tenkan_sen = (nine_period_high + nine_period_low) / 2
        
        twenty_six_period_high = ohlc_df['High'].rolling(window=26).max()
        twenty_six_period_low = ohlc_df['Low'].rolling(window=26).min()
        kijun_sen = (twenty_six_period_high + twenty_six_period_low) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        fifty_two_period_high = ohlc_df['High'].rolling(window=52).max()
        fifty_two_period_low = ohlc_df['Low'].rolling(window=52).min()
        senkou_span_b = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)
        
        chikou_span = ohlc_df['Close'].shift(-26)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=senkou_span_a, mode='lines', name='Senkou A', line=dict(color='#22c55e', width=1), fill='tozeroy', fillcolor="rgba(34, 197, 94, 0.2)"))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=senkou_span_b, mode='lines', name='Senkou B', line=dict(color='#ef4444', width=1), fill='tonexty', fillcolor="rgba(239, 68, 68, 0.2)"))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=tenkan_sen, mode='lines', name='Tenkan Sen', line=dict(color='#f59e0b', width=2)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=kijun_sen, mode='lines', name='Kijun Sen', line=dict(color='#8b5cf6', width=2)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df['Close'], mode='lines', name='Close', line=dict(color='#f8fafc', width=1.5)))
        
        fig.update_layout(
            title=dict(text=f'<b>Ichimoku Cloud: {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Price', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=450, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_pivot_points(self, ohlc_df, ticker):
        """Pivot Points - Support and resistance levels based on previous period prices."""
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        
        pivot = (ohlc_df['High'].shift(1) + ohlc_df['Low'].shift(1) + ohlc_df['Close'].shift(1)) / 3
        r1 = 2 * pivot - ohlc_df['Low'].shift(1)
        s1 = 2 * pivot - ohlc_df['High'].shift(1)
        r2 = pivot + (ohlc_df['High'].shift(1) - ohlc_df['Low'].shift(1))
        s2 = pivot - (ohlc_df['High'].shift(1) - ohlc_df['Low'].shift(1))
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=ohlc_df.index, open=ohlc_df['Open'], high=ohlc_df['High'], low=ohlc_df['Low'], close=ohlc_df['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=r2, mode='lines', name='R2', line=dict(color='#ef4444', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=r1, mode='lines', name='R1', line=dict(color='#ef4444', width=1.5)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=pivot, mode='lines', name='Pivot', line=dict(color='#fbbf24', width=2)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=s1, mode='lines', name='S1', line=dict(color='#22c55e', width=1.5)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=s2, mode='lines', name='S2', line=dict(color='#22c55e', width=1, dash='dot')))
        
        fig.update_layout(
            title=dict(text=f'<b>Pivot Points: {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Price', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=450, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_ease_of_movement(self, ohlc_df, ticker, period=14):
        """Ease of Movement - Relates price change to volume."""
        if ohlc_df is None or ohlc_df.empty or 'Volume' not in ohlc_df.columns:
            return go.Figure()
        
        distance = ((ohlc_df['High'] + ohlc_df['Low']) / 2) - ((ohlc_df['High'].shift(1) + ohlc_df['Low'].shift(1)) / 2)
        box_ratio = (ohlc_df['Volume'] / 1000000) / (ohlc_df['High'] - ohlc_df['Low']).replace(0, np.nan)
        eom = distance.rolling(window=period).mean() / box_ratio.rolling(window=period).mean()
        
        colors = np.where(eom >= 0, '#22c55e', '#ef4444')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ohlc_df.index, y=eom, marker_color=colors, name='EoM'))
        fig.add_hline(y=0, line_dash='dot', line_color='#64748b')
        
        fig.update_layout(
            title=dict(text=f'<b>Ease of Movement ({period}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='EoM', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=350, margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_force_index(self, ohlc_df, ticker, period=13):
        """Force Index - Combines price change and volume to measure force."""
        if ohlc_df is None or ohlc_df.empty or 'Volume' not in ohlc_df.columns:
            return go.Figure()
        
        force = (ohlc_df['Close'] - ohlc_df['Close'].shift(1)) * ohlc_df['Volume']
        force_ema = force.ewm(span=period).mean()
        
        colors = np.where(force >= 0, '#22c55e', '#ef4444')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ohlc_df.index, y=force, marker_color=colors, name='Force', opacity=0.5))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=force_ema, mode='lines', name='Force EMA', line=dict(color='#8b5cf6', width=2.5)))
        fig.add_hline(y=0, line_dash='dot', line_color='#64748b')
        
        fig.update_layout(
            title=dict(text=f'<b>Force Index ({period}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Force', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=350, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_ultimate_oscillator(self, ohlc_df, ticker, period1=7, period2=14, period3=28):
        """Ultimate Oscillator - Combines multiple timeframes to reduce false signals."""
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        
        bp = ohlc_df['Close'] - pd.concat([ohlc_df['Low'], ohlc_df['Close'].shift(1)], axis=1).min(axis=1)
        tr = pd.concat([ohlc_df['High'], ohlc_df['Close'].shift(1)], axis=1).max(axis=1) - pd.concat([ohlc_df['Low'], ohlc_df['Close'].shift(1)], axis=1).min(axis=1)
        
        avg1 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
        avg2 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
        avg3 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()
        uo = 100 * ((4 * avg1 + 2 * avg2 + avg3) / 7)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=uo, mode='lines', name='Ultimate Osc', line=dict(color='#14b8a6', width=2)))
        fig.add_hline(y=70, line_dash='dash', line_color='#ef4444', annotation_text='Overbought')
        fig.add_hline(y=30, line_dash='dash', line_color='#22c55e', annotation_text='Oversold')
        
        fig.update_layout(
            title=dict(text=f'<b>Ultimate Oscillator ({period1},{period2},{period3}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='UO', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            yaxis=dict(range=[0, 100]), height=350,
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_aroon_indicator(self, ohlc_df, ticker, period=25):
        """Aroon Indicator - Identifies trend changes and strength."""
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        
        aroon_up = ohlc_df['High'].rolling(window=period + 1).apply(lambda x: float(np.argmax(x)) / period * 100, raw=True)
        aroon_down = ohlc_df['Low'].rolling(window=period + 1).apply(lambda x: float(np.argmin(x)) / period * 100, raw=True)
        aroon_oscillator = aroon_up - aroon_down
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=aroon_up, mode='lines', name='Aroon Up', line=dict(color='#22c55e', width=2)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=aroon_down, mode='lines', name='Aroon Down', line=dict(color='#ef4444', width=2)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=aroon_oscillator, mode='lines', name='Oscillator', line=dict(color='#8b5cf6', width=1.5)))
        fig.add_hline(y=0, line_dash='dot', line_color='#64748b')
        
        fig.update_layout(
            title=dict(text=f'<b>Aroon Indicator ({period}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Aroon', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            yaxis=dict(range=[0, 100]), height=350,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_chaikin_oscillator(self, ohlc_df, ticker, fast=3, slow=10):
        """Chaikin Oscillator - Measures momentum of Accumulation/Distribution line."""
        if ohlc_df is None or ohlc_df.empty or 'Volume' not in ohlc_df.columns:
            return go.Figure()
        
        clv = ((ohlc_df['Close'] - ohlc_df['Low']) - (ohlc_df['High'] - ohlc_df['Close'])) / (ohlc_df['High'] - ohlc_df['Low']).replace(0, np.nan)
        ad = clv * ohlc_df['Volume']
        ad_line = ad.cumsum()
        fast_ma = ad_line.ewm(span=fast).mean()
        slow_ma = ad_line.ewm(span=slow).mean()
        chaikin = fast_ma - slow_ma
        
        colors = np.where(chaikin >= 0, '#22c55e', '#ef4444')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ohlc_df.index, y=chaikin, marker_color=colors, name='Chaikin Osc'))
        fig.add_hline(y=0, line_dash='dot', line_color='#64748b')
        
        fig.update_layout(
            title=dict(text=f'<b>Chaikin Oscillator ({fast},{slow}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Chaikin', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=350, margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_demand_index(self, ohlc_df, ticker, period=14):
        """Demand Index - Combines price and volume to identify supply/demand balance."""
        if ohlc_df is None or ohlc_df.empty or 'Volume' not in ohlc_df.columns:
            return go.Figure()
        
        clv = ((ohlc_df['Close'] - ohlc_df['Low']) - (ohlc_df['High'] - ohlc_df['Close'])) / (ohlc_df['High'] - ohlc_df['Low']).replace(0, np.nan)
        demand = clv * ohlc_df['Volume']
        demand_ma = demand.rolling(window=period).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ohlc_df.index, y=demand, marker_color='#3b82f6', name='Demand', opacity=0.6))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=demand_ma, mode='lines', name=f'Demand MA ({period})', line=dict(color='#f59e0b', width=2.5)))
        fig.add_hline(y=0, line_dash='dot', line_color='#64748b')
        
        fig.update_layout(
            title=dict(text=f'<b>Demand Index ({period}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Demand', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=350, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_mass_index(self, ohlc_df, ticker, period1=9, period2=25):
        """Mass Index - Identifies trend reversal by analyzing narrowing/widening of price range."""
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        
        hl_range = ohlc_df['High'] - ohlc_df['Low']
        ema1 = hl_range.ewm(span=period1).mean()
        ema2 = ema1.ewm(span=period1).mean()
        mass = ema2 / ema2.ewm(span=period1).mean()
        mass_index = mass.rolling(window=period2).sum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=mass_index, mode='lines', name='Mass Index', line=dict(color='#f472b6', width=2)))
        fig.add_hline(y=27, line_dash='dash', line_color='#ef4444', annotation_text='Trend Change')
        fig.add_hline(y=26.5, line_dash='dash', line_color='#22c55e', annotation_text='Trend Change')
        
        fig.update_layout(
            title=dict(text=f'<b>Mass Index ({period1},{period2}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Mass Index', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=350, margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_true_strength_index(self, ohlc_df, ticker, period1=25, period2=13, period3=13):
        """True Strength Index (TSI) - Momentum oscillator using double smoothed momentum."""
        if ohlc_df is None or ohlc_df.empty or 'Close' not in ohlc_df.columns:
            return go.Figure()
        
        momentum = ohlc_df['Close'].diff()
        abs_momentum = np.abs(momentum)
        
        tsi = (momentum.ewm(span=period1).mean().ewm(span=period2).mean() / 
               abs_momentum.ewm(span=period1).mean().ewm(span=period3).mean()) * 100
        
        colors = np.where(tsi >= 0, '#22c55e', '#ef4444')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ohlc_df.index, y=tsi, marker_color=colors, name='TSI'))
        fig.add_hline(y=0, line_dash='dot', line_color='#64748b')
        
        fig.update_layout(
            title=dict(text=f'<b>True Strength Index ({period1},{period2},{period3}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='TSI', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=350, margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_kst_indicator(self, ohlc_df, ticker):
        """KST (Know Sure Thing) - Momentum oscillator based on smoothed ROC."""
        if ohlc_df is None or ohlc_df.empty or 'Close' not in ohlc_df.columns:
            return go.Figure()
        
        def get_kst(periods):
            roc = ((ohlc_df['Close'] - ohlc_df['Close'].shift(periods[0])) / ohlc_df['Close'].shift(periods[0]))
            return roc.rolling(window=periods[1]).mean()
        
        kst = (get_kst([10, 10]) * 1 + get_kst([15, 10]) * 2 + get_kst([20, 10]) * 3 + get_kst([30, 15]) * 4) / 10
        kst_signal = kst.rolling(window=9).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=kst, mode='lines', name='KST', line=dict(color='#3b82f6', width=2)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=kst_signal, mode='lines', name='Signal', line=dict(color='#f59e0b', width=2)))
        fig.add_hline(y=0, line_dash='dot', line_color='#64748b')
        
        fig.update_layout(
            title=dict(text=f'<b>KST Indicator: {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='KST', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=350, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_accumulation_distribution(self, ohlc_df, ticker):
        """Accumulation/Distribution Line - Shows cumulative money flow."""
        if ohlc_df is None or ohlc_df.empty or 'Volume' not in ohlc_df.columns:
            return go.Figure()
        
        clv = ((ohlc_df['Close'] - ohlc_df['Low']) - (ohlc_df['High'] - ohlc_df['Close'])) / (ohlc_df['High'] - ohlc_df['Low']).replace(0, np.nan)
        ad = (clv * ohlc_df['Volume']).cumsum()
        ad_ma = ad.rolling(window=21).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ad, mode='lines', name='A/D Line', line=dict(color='#8b5cf6', width=2)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ad_ma, mode='lines', name='21-day MA', line=dict(color='#f59e0b', width=1.5)))
        
        fig.update_layout(
            title=dict(text=f'<b>Accumulation/Distribution: {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='A/D', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=400, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8', tickformat=',d')
        return fig

    def plot_volume_profile(self, ohlc_df, ticker, bins=20):
        """Volume Profile - Shows trading activity at different price levels."""
        if ohlc_df is None or ohlc_df.empty or 'Volume' not in ohlc_df.columns:
            return go.Figure()
        
        price_bins = pd.cut(ohlc_df['Close'], bins=bins)
        volume_profile = ohlc_df.groupby(price_bins)['Volume'].sum()
        
        fig = go.Figure(go.Bar(
            x=volume_profile.values,
            y=[f'{i.left:.2f}-{i.right:.2f}' for i in volume_profile.index],
            orientation='h',
            marker_color='#3b82f6',
            name='Volume'
        ))
        
        fig.update_layout(
            title=dict(text=f'<b>Volume Profile: {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Volume', yaxis_title='Price Range', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=450, margin=dict(l=80, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8', tickformat=',d')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_price_volume_trend(self, ohlc_df, ticker):
        """Price Volume Trend - Combines price change and volume direction."""
        if ohlc_df is None or ohlc_df.empty or 'Volume' not in ohlc_df.columns:
            return go.Figure()
        
        pvt = ((ohlc_df['Close'] - ohlc_df['Close'].shift(1)) / ohlc_df['Close'].shift(1)) * ohlc_df['Volume']
        pvt = pvt.cumsum()
        pvt_ma = pvt.rolling(window=21).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=pvt, mode='lines', name='PVT', line=dict(color='#10b981', width=2)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=pvt_ma, mode='lines', name='21-day MA', line=dict(color='#f59e0b', width=1.5)))
        
        fig.update_layout(
            title=dict(text=f'<b>Price Volume Trend: {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='PVT', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=400, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8', tickformat=',d')
        return fig

    def plot_heikin_ashi_candles(self, ohlc_df, ticker):
        """Heikin Ashi Candles - Smoothed candlestick chart to identify trends."""
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        
        ha_close = (ohlc_df['Open'] + ohlc_df['High'] + ohlc_df['Low'] + ohlc_df['Close']) / 4
        ha_open = (ohlc_df['Open'].shift(1) + ohlc_df['Close'].shift(1)) / 2
        ha_open.iloc[0] = ohlc_df['Open'].iloc[0]
        ha_high = pd.concat([ohlc_df['High'], ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([ohlc_df['Low'], ha_open, ha_close], axis=1).min(axis=1)
        
        colors = np.where(ha_close >= ha_open, '#22c55e', '#ef4444')
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=ohlc_df.index, open=ha_open, high=ha_high, low=ha_low, close=ha_close, name='Heikin Ashi'))
        
        fig.update_layout(
            title=dict(text=f'<b>Heikin Ashi: {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Price', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=450, margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_renko_chart(self, ohlc_df, ticker, brick_size=None):
        """Renko Chart - Price-based chart filtering minor movements."""
        if ohlc_df is None or ohlc_df.empty or 'Close' not in ohlc_df.columns:
            return go.Figure()
        
        if brick_size is None:
            brick_size = ohlc_df['Close'].std() * 2
        
        renko_data = []
        current_price = ohlc_df['Close'].iloc[0]
        trend = None
        
        for close in ohlc_df['Close']:
            if trend is None:
                if close > current_price + brick_size:
                    trend = 1
                    renko_data.append({'date': ohlc_df.index[ohlc_df['Close'] == close][0], 'price': close, 'direction': 1})
                    current_price = close
                elif close < current_price - brick_size:
                    trend = -1
                    renko_data.append({'date': ohlc_df.index[ohlc_df['Close'] == close][0], 'price': close, 'direction': -1})
                    current_price = close
            else:
                if trend == 1 and close > current_price + brick_size:
                    renko_data.append({'date': ohlc_df.index[ohlc_df['Close'] == close][0], 'price': close, 'direction': 1})
                    current_price = close
                elif trend == -1 and close < current_price - brick_size:
                    renko_data.append({'date': ohlc_df.index[ohlc_df['Close'] == close][0], 'price': close, 'direction': -1})
                    current_price = close
        
        if not renko_data:
            return go.Figure()
        
        renko_df = pd.DataFrame(renko_data)
        colors = np.where(renko_df['direction'] == 1, '#22c55e', '#ef4444')
        
        fig = go.Figure(go.Bar(x=renko_df['date'], y=renko_df['direction'], marker_color=colors))
        
        fig.update_layout(
            title=dict(text=f'<b>Renko Chart (size={brick_size:.2f}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Direction', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=350, margin=dict(l=60, r=40, t=80, b=40),
            yaxis=dict(tickvals=[-1, 1], ticktext=['Down', 'Up'])
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_keltner_channels(self, ohlc_df, ticker, period=20, mult=2):
        """Keltner Channels - Volatility-based channels using ATR."""
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        
        ema = ohlc_df['Close'].ewm(span=period).mean()
        high_low = ohlc_df['High'] - ohlc_df['Low']
        high_close = np.abs(ohlc_df['High'] - ohlc_df['Close'].shift())
        low_close = np.abs(ohlc_df['Low'] - ohlc_df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        upper = ema + (mult * atr)
        lower = ema - (mult * atr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df['Close'], mode='lines', name='Close', line=dict(color='#f8fafc', width=1.5)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=upper, mode='lines', name='Upper', line=dict(color='#ef4444', width=1)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ema, mode='lines', name='EMA', line=dict(color='#fbbf24', width=2)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=lower, mode='lines', name='Lower', line=dict(color='#22c55e', width=1), fill='tonexty', fillcolor='rgba(34, 197, 94, 0.1)'))
        
        fig.update_layout(
            title=dict(text=f'<b>Keltner Channels ({period},{mult}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Price', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=400, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig

    def plot_parabolic_sar(self, ohlc_df, ticker, acceleration=0.02, maximum=0.2):
        """Parabolic SAR - Trend following indicator showing potential reversal points."""
        if ohlc_df is None or ohlc_df.empty:
            return go.Figure()
        
        sar = pd.Series(index=ohlc_df.index, dtype=float)
        trend = 1
        af = acceleration
        ep = ohlc_df['High'].iloc[0]
        sar.iloc[0] = ohlc_df['Low'].iloc[0]
        
        for i in range(1, len(ohlc_df)):
            if trend == 1:
                sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                if ohlc_df['Low'].iloc[i] < sar.iloc[i]:
                    trend = -1
                    sar.iloc[i] = ep
                    ep = ohlc_df['Low'].iloc[i]
                    af = acceleration
                else:
                    if ohlc_df['High'].iloc[i] > ep:
                        ep = ohlc_df['High'].iloc[i]
                        af = min(af + acceleration, maximum)
            else:
                sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                if ohlc_df['High'].iloc[i] > sar.iloc[i]:
                    trend = 1
                    sar.iloc[i] = ep
                    ep = ohlc_df['High'].iloc[i]
                    af = acceleration
                else:
                    if ohlc_df['Low'].iloc[i] < ep:
                        ep = ohlc_df['Low'].iloc[i]
                        af = min(af + acceleration, maximum)
        
        colors = np.where(ohlc_df['Close'] >= sar, '#22c55e', '#ef4444')
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=ohlc_df.index, open=ohlc_df['Open'], high=ohlc_df['High'], low=ohlc_df['Low'], close=ohlc_df['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=sar, mode='markers', name='SAR', marker=dict(color='#f59e0b', size=6, symbol='circle')))
        
        fig.update_layout(
            title=dict(text=f'<b>Parabolic SAR (af={acceleration}, max={maximum}): {ticker}</b>', font=dict(size=18, color='#f8fafc')),
            xaxis_title='Date', yaxis_title='Price', template='plotly_dark',
            paper_bgcolor='#0f172a', plot_bgcolor='#111827', font=dict(color='#f8fafc'),
            height=450, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=40, t=80, b=40)
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148, 163, 184, 0.15)', color='#94a3b8')
        return fig
