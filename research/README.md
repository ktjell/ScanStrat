# research/

Scripts til kvantitativ research. Bruger `engine/` som bibliotek.

## Kør et script

```
uv run python research/run_backtest.py
uv run python research/run_backtest.py --universe EU --top-n 5 --start 2022-01-01

uv run python research/compare_strategies.py
uv run python research/compare_strategies.py --universe US --top-n 10

uv run python research/run_parameter_search.py
uv run python research/run_parameter_search.py --metric sharpe --top 20

uv run python research/run_reversal_experiment.py
uv run python research/run_reversal_experiment.py --rsi 30 --vol 1.5 --lookback 30
```

## Scripts

| Script | Formål |
|---|---|
| `run_backtest.py` | Walk-forward backtest, brutto vs. netto |
| `compare_strategies.py` | Momentum vs. Reversal side om side |
| `run_parameter_search.py` | Grid search over RSI / volumen / lookback |
| `run_reversal_experiment.py` | Dybdegående reversal-analyse + screening |
