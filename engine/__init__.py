"""Engine package — re-exports the core library modules.

Import from here in research scripts to keep the API stable:

    from engine.features import Momentum12M, RSI14
    from engine.ranking import Ranker, ReversalStrategy
    from engine.backtest import BacktestEngine, CommissionSchedule
    from engine.data import DataService
    from engine.config import Settings, ReversalSettings
"""
