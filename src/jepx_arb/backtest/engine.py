"""Run the dispatch optimiser across a date range and total up the result.

Two horizons, and the difference matters:

``day``
    Each delivery day is solved on its own and the battery ends where it
    started.  This mirrors how the day-ahead market actually works -- you
    commit a schedule one day at a time -- and it is the default.

``continuous``
    One LP over the whole period, SOC carrying across midnight.  Strictly
    higher than the daily figure, because it can hold charge for days
    waiting for a better spread.  Useful as an outer bound; not something
    anyone can operate against.

Either way the number is an upper bound with perfect hindsight.  A real
strategy running on forecasts captures a fraction of it, and that fraction
is the thing worth optimising later.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from jepx_arb.backtest.dispatch import DispatchResult, optimise
from jepx_arb.battery import BatterySpec
from jepx_arb.load import for_date as load_for_date
from jepx_arb.tariff import Tariff
from jepx_arb.timeslot import SLOTS_PER_DAY, slot_hour_of_day

log = logging.getLogger(__name__)

# Fixed so that an empty run still yields a frame the summary can total up:
# a window with no complete days must report zeroes, not raise.
DAILY_COLUMNS: tuple[str, ...] = (
    "delivery_date",
    "net_profit_jpy",
    "revenue_jpy",
    "energy_cost_jpy",
    "degradation_jpy",
    "charged_kwh",
    "discharged_kwh",
    "cycles",
    "market_min",
    "market_max",
    "market_mean",
    "market_spread",
    "soc_end_kwh",
)


@dataclass
class BacktestConfig:
    """Everything one run needs beyond the price series itself."""

    spec: BatterySpec
    tariff: Tariff
    area: str = "tokyo"
    horizon: str = "day"
    terminal: str = "initial"

    load_profile: str | None = None
    """``household`` / ``commercial`` / ``flat``; required when the battery
    cannot export, because then the load is what caps discharge."""

    load_daily_kwh: float = 12.0
    load_series: pd.Series | None = None
    """Real half-hourly metering indexed by ``(delivery_date, slot)``,
    which overrides the synthetic profile when present."""

    keep_schedule: bool = False
    """Retain the per-slot dispatch.  A year is ~17.5k rows per area, so
    this is opt-in rather than always on."""

    def __post_init__(self) -> None:
        if self.horizon not in ("day", "continuous"):
            raise ValueError(f"horizon must be 'day' or 'continuous', got {self.horizon!r}")
        if not self.spec.allow_export and self.load_profile is None and self.load_series is None:
            raise ValueError(
                f"{self.spec.name} cannot export, so discharge is capped by site load: "
                "set load_profile (household/commercial/flat) or pass load_series"
            )


@dataclass
class BacktestResult:
    """Per-day results plus the totals worth quoting."""

    config: BacktestConfig
    daily: pd.DataFrame
    schedule: pd.DataFrame | None = None
    skipped_days: list[tuple[dt.date, str]] = field(default_factory=list)

    @property
    def days(self) -> int:
        return len(self.daily)

    @property
    def net_profit_jpy(self) -> float:
        return float(self.daily["net_profit_jpy"].sum())

    @property
    def revenue_jpy(self) -> float:
        return float(self.daily["revenue_jpy"].sum())

    @property
    def energy_cost_jpy(self) -> float:
        return float(self.daily["energy_cost_jpy"].sum())

    @property
    def degradation_jpy(self) -> float:
        return float(self.daily["degradation_jpy"].sum())

    @property
    def discharged_kwh(self) -> float:
        return float(self.daily["discharged_kwh"].sum())

    @property
    def charged_kwh(self) -> float:
        return float(self.daily["charged_kwh"].sum())

    @property
    def equivalent_full_cycles(self) -> float:
        return self.discharged_kwh / self.config.spec.ac_kwh_per_full_cycle

    @property
    def annualised_profit_jpy(self) -> float:
        """Net profit scaled to 365 days -- the only fair way to compare
        runs over different length windows."""
        return self.net_profit_jpy / self.days * 365 if self.days else 0.0

    @property
    def profit_per_kwh_capacity_year(self) -> float:
        """円/kWh of nameplate capacity per year: the number that lets a
        home battery and a grid-scale system sit in the same table."""
        return self.annualised_profit_jpy / self.config.spec.capacity_kwh

    @property
    def active_day_share(self) -> float:
        """Fraction of days the battery moved at all.  A low number means
        the wear cost is holding it back, not the price shape."""
        if not self.days:
            return 0.0
        return float((self.daily["discharged_kwh"] > 1e-6).mean())


def run_backtest(prices: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """Backtest ``config`` against long-format prices for a single area.

    ``prices`` needs ``delivery_date``, ``slot`` and ``price_jpy_kwh``; an
    ``area`` column is filtered on when present.
    """
    frame = _prepare(prices, config.area)
    if config.horizon == "continuous":
        return _run_continuous(frame, config)
    return _run_daily(frame, config)


def _run_daily(frame: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    hours = slot_hour_of_day()
    rows: list[dict] = []
    schedules: list[pd.DataFrame] = []
    skipped: list[tuple[dt.date, str]] = []
    soc = config.spec.soc_initial_kwh

    for date, day in frame.groupby("delivery_date", sort=True):
        market = _day_prices(day)
        if market is None:
            skipped.append((date, _invalid_day_reason(day)))
            continue

        buy, sell = config.tariff.prices(market, hours)
        result = optimise(
            config.spec,
            buy,
            sell,
            soc_start_kwh=soc,
            max_discharge_kwh=_discharge_cap(config, date),
            terminal=config.terminal,
        )
        soc = float(result.soc_kwh[-1])
        rows.append(_summarise_day(date, market, result, config.spec))
        if config.keep_schedule:
            schedules.append(_schedule_rows(date, market, result))

    daily = _daily_frame(rows)
    schedule = pd.concat(schedules, ignore_index=True) if schedules else None
    if skipped:
        log.warning("skipped %d incomplete day(s); first: %s", len(skipped), skipped[0])
    return BacktestResult(config=config, daily=daily, schedule=schedule, skipped_days=skipped)


def _run_continuous(frame: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    complete: list[pd.DataFrame] = []
    skipped: list[tuple[dt.date, str]] = []
    for date, day in frame.groupby("delivery_date", sort=True):
        if _day_prices(day) is None:
            skipped.append((date, _invalid_day_reason(day)))
            continue
        complete.append(day.sort_values("slot"))
    if not complete:
        return BacktestResult(config=config, daily=_daily_frame([]), skipped_days=skipped)

    joined = pd.concat(complete, ignore_index=True)
    dates = list(dict.fromkeys(joined["delivery_date"]))
    market = joined["price_jpy_kwh"].to_numpy(dtype=float)
    hours = np.tile(slot_hour_of_day(), len(dates))
    buy, sell = config.tariff.prices(market, hours)

    caps = None
    if not config.spec.allow_export:
        caps = np.concatenate([_discharge_cap(config, date) for date in dates])

    result = optimise(
        config.spec,
        buy,
        sell,
        max_discharge_kwh=caps,
        terminal=config.terminal,
        cycle_cap=(
            None
            if config.spec.max_cycles_per_day is None
            else config.spec.max_cycles_per_day * len(dates)
        ),
    )

    rows = []
    schedules = []
    for index, date in enumerate(dates):
        window = slice(index * SLOTS_PER_DAY, (index + 1) * SLOTS_PER_DAY)
        day_result = DispatchResult(
            charge_kwh=result.charge_kwh[window],
            discharge_kwh=result.discharge_kwh[window],
            soc_kwh=result.soc_kwh[window],
            buy_price=result.buy_price[window],
            sell_price=result.sell_price[window],
            degradation_cost_jpy_per_kwh=result.degradation_cost_jpy_per_kwh,
        )
        rows.append(_summarise_day(date, market[window], day_result, config.spec))
        if config.keep_schedule:
            schedules.append(_schedule_rows(date, market[window], day_result))

    return BacktestResult(
        config=config,
        daily=_daily_frame(rows),
        schedule=pd.concat(schedules, ignore_index=True) if schedules else None,
        skipped_days=skipped,
    )


def _daily_frame(rows: list[dict]) -> pd.DataFrame:
    """Per-day results, always carrying the full column set."""
    return pd.DataFrame(rows, columns=list(DAILY_COLUMNS))


def _prepare(prices: pd.DataFrame, area: str) -> pd.DataFrame:
    frame = prices
    if "area" in frame.columns:
        frame = frame[frame["area"] == area]
        if frame.empty:
            available = sorted(prices["area"].unique()) if "area" in prices else []
            raise ValueError(f"no rows for area {area!r}; available: {available}")
    missing = {"delivery_date", "slot", "price_jpy_kwh"} - set(frame.columns)
    if missing:
        raise ValueError(f"prices is missing column(s): {sorted(missing)}")
    return frame.sort_values(["delivery_date", "slot"])


def _day_prices(day: pd.DataFrame) -> np.ndarray | None:
    """Prices as a dense 48-vector, or None if the day is incomplete."""
    if len(day) != SLOTS_PER_DAY:
        return None
    slots = pd.to_numeric(day["slot"], errors="coerce").sort_values().to_numpy()
    if not np.array_equal(slots, np.arange(1, SLOTS_PER_DAY + 1)):
        return None
    ordered = day.sort_values("slot")["price_jpy_kwh"].to_numpy(dtype=float)
    if not np.isfinite(ordered).all():
        return None
    return ordered


def _invalid_day_reason(day: pd.DataFrame) -> str:
    """Describe invalid input without implying that row count alone is sufficient."""
    unique_slots = day["slot"].nunique(dropna=True)
    return (
        f"expected slots 1..{SLOTS_PER_DAY} exactly once with finite prices; "
        f"found {len(day)} rows and {unique_slots} unique slots"
    )


def _discharge_cap(config: BacktestConfig, date: dt.date) -> np.ndarray | None:
    if config.spec.allow_export:
        return None
    if config.load_series is not None:
        from jepx_arb.load import from_series

        return from_series(config.load_series, date)
    return load_for_date(config.load_profile or "household", date, config.load_daily_kwh)


def _summarise_day(
    date: dt.date, market: np.ndarray, result: DispatchResult, spec: BatterySpec
) -> dict:
    return {
        "delivery_date": date,
        "net_profit_jpy": result.net_profit_jpy,
        "revenue_jpy": result.revenue_jpy,
        "energy_cost_jpy": result.energy_cost_jpy,
        "degradation_jpy": result.degradation_jpy,
        "charged_kwh": result.charged_kwh,
        "discharged_kwh": result.discharged_kwh,
        "cycles": result.equivalent_full_cycles(spec),
        "market_min": float(market.min()),
        "market_max": float(market.max()),
        "market_mean": float(market.mean()),
        "market_spread": float(market.max() - market.min()),
        "soc_end_kwh": float(result.soc_kwh[-1]),
    }


def _schedule_rows(date: dt.date, market: np.ndarray, result: DispatchResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "delivery_date": date,
            "slot": np.arange(1, len(market) + 1),
            "market_jpy_kwh": market,
            "buy_jpy_kwh": result.buy_price,
            "sell_jpy_kwh": result.sell_price,
            "charge_kwh": result.charge_kwh,
            "discharge_kwh": result.discharge_kwh,
            "soc_kwh": result.soc_kwh,
        }
    )
