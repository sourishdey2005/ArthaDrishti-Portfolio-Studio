# Black-Litterman Portfolio Studio

**Black-Litterman Portfolio Studio** is an interactive Python web application that combines the Black-Litterman model with real-time market data, investor views, and rich Plotly visualizations to construct and analyze optimized equity portfolios.

## Key Capabilities
- **Real-time data ingestion** via Yahoo Finance for equity OHLCV history and market capitalization.
- **Investor views management**: add absolute or relative opinions, assign confidence levels, and see their impact on the posterior returns.
- **Black-Litterman posterior** generation that blends equilibrium market views with custom investor convictions.
- **Portfolio optimization** using mean-variance techniques, including efficient frontier computation and constraint handling.
- **Interactive visualizations** powered by Plotly to explore returns, risk, portfolio weights, Donchian/Keltner channels, and other indicators.
- **Results export** to CSV for downstream analysis.

## Folder Structure
- `app.py`: Streamlit entrypoint that wires together the UI, data loading, and visualization helpers.
- `config/settings.py`: Centralized settings for API credentials, visualization styling, and default parameters.
- `data/`: Data ingestion utilities (e.g., `yahoo_data.py`) and supporting assets.
- `models/`: Black-Litterman math, optimization routines, and related helpers.
- `utils/`: Visualization helpers (`visualization.py`, `fix_viz.py`), user-facing formatting utilities, and notebook-ready helpers.

## Getting Started
### Prerequisites
- Python 3.11
- Git for repository management
- Recommended: create a virtual environment (`python -m venv .venv`) to isolate dependencies.

### Installation
```bash
git clone https://github.com/sourishdey2005/ArthaDrishti-Portfolio-Studio.git
cd black_litterman_app
python -m venv .venv
.\.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Configuration
1. Review `config/settings.py` to adjust defaults:
   - `MARKET_DATABASE`, `VIEW_CONFIDENCE_PRIOR`, and color palettes.
2. Place any API keys or secrets in a `.env` file (ensure `.env` is listed in `.gitignore`).
3. Feed custom data by dropping CSVs into `data/` and hooking them into `app.py` if needed.

### Running the Application
```bash
streamlit run app.py
```
- Navigate to the shown URL (typically `http://localhost:8501`).
- Use the sidebar to load tickers, compose views, and trigger portfolio recalculations.

## Development & Testing
- Format code with `black` and `ruff` if installed.
- Unit tests: none currently shipped; consider writing coverage for optimization logic before refactoring.
- When adding new visualization modules, keep Plotly configuration in `utils/visualization.py`.
- Use `git status` to confirm staged changes and `git commit` with descriptive summaries.

## Deployment Notes
- Streamlit apps can be deployed via Streamlit Cloud, AWS, or Docker containers.
- Package requirements are maintained in `requirements.txt`; update as dependencies evolve.
- Secure any secrets using environment variables or a secrets manager in production.

## Contribution Guidelines
1. Fork the repo and create a feature branch (`feature/your-topic`).
2. Follow the prevailing code style (PEP 8 / Streamlit-friendly).
3. Test locally before submitting a pull request.
4. Describe the change succinctly in your PR and reference any issue it addresses.

## License & Contact
- License: MIT (see `LICENSE` from the upstream repository).
- Maintained by Sourish Dey; reach out via the GitHub repository for questions or support.

## References
- [Black-Litterman model overview](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=8443)
- [Plotly documentation](https://plotly.com/python/)
