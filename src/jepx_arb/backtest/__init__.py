"""Perfect-foresight backtesting: what the battery *could* have earned."""

from jepx_arb.backtest.dispatch import DispatchResult, optimise
from jepx_arb.backtest.engine import BacktestConfig, BacktestResult, run_backtest

__all__ = [
    "DispatchResult",
    "optimise",
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
]
