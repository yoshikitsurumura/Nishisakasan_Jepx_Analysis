"""Turn a backtest into something you can put in front of a partner.

The headline is deliberately 円/kWh of capacity per year rather than total
yen: it is the only figure on which an 8 kWh home unit and a 2 MWh
grid-scale system can be compared without mental arithmetic.

Every report carries the perfect-foresight caveat.  A number produced here
is a ceiling, not a forecast, and it stops being honest the moment it is
quoted without that sentence attached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jepx_arb.backtest.engine import BacktestResult
from jepx_arb.sources.jepx_csv import ATTRIBUTION
from jepx_arb.timeslot import SLOTS_PER_DAY, slot_hour_of_day

CEILING_NOTE = (
    "Perfect foresight: this is the ceiling a flawless forecast would have hit, "
    "not an expected return."
)


def summarise(result: BacktestResult) -> dict:
    """The headline numbers as a flat dict, for tables and JSON alike."""
    spec = result.config.spec
    daily = result.daily
    first = daily["delivery_date"].min() if len(daily) else None
    last = daily["delivery_date"].max() if len(daily) else None
    return {
        "preset": spec.name,
        "area": result.config.area,
        "tariff": result.config.tariff.name,
        "horizon": result.config.horizon,
        "date_from": first,
        "date_to": last,
        "days": result.days,
        "capacity_kwh": spec.capacity_kwh,
        "power_kw": spec.power_kw,
        "wear_jpy_per_kwh": spec.degradation_cost_jpy_per_kwh,
        "revenue_jpy": result.revenue_jpy,
        "energy_cost_jpy": result.energy_cost_jpy,
        "degradation_jpy": result.degradation_jpy,
        "net_profit_jpy": result.net_profit_jpy,
        "annualised_jpy": result.annualised_profit_jpy,
        "jpy_per_kwh_capacity_year": result.profit_per_kwh_capacity_year,
        "charged_kwh": result.charged_kwh,
        "discharged_kwh": result.discharged_kwh,
        "equivalent_full_cycles": result.equivalent_full_cycles,
        "cycles_per_day": result.equivalent_full_cycles / result.days if result.days else 0.0,
        "active_day_share": result.active_day_share,
        "mean_daily_spread_jpy": float(daily["market_spread"].mean()) if len(daily) else 0.0,
        "skipped_days": len(result.skipped_days),
    }


def format_report(result: BacktestResult, top_days: int = 5) -> str:
    """A human-readable block for the terminal."""
    stats = summarise(result)
    spec = result.config.spec
    if not result.days:
        return "No complete delivery days in range -- nothing to report."

    mean_buy = _mean_buy_price(result)
    lines = [
        "=" * 72,
        f"{stats['preset']}  |  area: {stats['area']}  |  tariff: {stats['tariff']}",
        f"{stats['date_from']} .. {stats['date_to']}  ({stats['days']} days, "
        f"{stats['horizon']} horizon)",
        "=" * 72,
        "",
        "System",
        f"  capacity                {spec.capacity_kwh:,.0f} kWh "
        f"({spec.usable_kwh:,.1f} kWh usable)",
        f"  power                   {spec.power_kw:,.0f} kW",
        f"  round-trip efficiency   {spec.round_trip_efficiency:.1%}",
        f"  wear cost               {spec.degradation_cost_jpy_per_kwh:,.2f} JPY/kWh discharged",
        f"  breakeven spread        {spec.breakeven_spread_jpy_kwh(mean_buy):,.2f} JPY/kWh "
        f"(at the {mean_buy:,.2f} JPY/kWh mean buy price)",
        f"  can export              {'yes' if spec.allow_export else 'no'}",
        "",
        "Result",
        f"  revenue                 {stats['revenue_jpy']:>14,.0f} JPY",
        f"  energy cost             {-stats['energy_cost_jpy']:>14,.0f} JPY",
        f"  wear                    {-stats['degradation_jpy']:>14,.0f} JPY",
        f"  {'net profit':<22}  {stats['net_profit_jpy']:>14,.0f} JPY",
        "",
        f"  annualised              {stats['annualised_jpy']:>14,.0f} JPY/year",
        f"  per kWh of capacity     {stats['jpy_per_kwh_capacity_year']:>14,.0f} "
        "JPY/kWh/year   <-- compare segments on this",
        "",
        "Utilisation",
        f"  discharged              {stats['discharged_kwh']:,.0f} kWh "
        f"({stats['equivalent_full_cycles']:,.1f} full cycles, "
        f"{stats['cycles_per_day']:.2f}/day)",
        f"  days it moved at all    {stats['active_day_share']:.1%}",
        f"  mean daily spread       {stats['mean_daily_spread_jpy']:,.2f} JPY/kWh",
    ]

    if stats["skipped_days"]:
        lines.append(
            f"  incomplete days skipped {stats['skipped_days']} "
            "(fewer than 48 published slots)"
        )

    if top_days:
        lines += ["", f"Best {top_days} days"]
        best = result.daily.nlargest(top_days, "net_profit_jpy")
        for _, row in best.iterrows():
            lines.append(
                f"  {row['delivery_date']}  {row['net_profit_jpy']:>10,.0f} JPY   "
                f"spread {row['market_spread']:>6.2f}   "
                f"{row['cycles']:.2f} cycles"
            )

    lines += ["", CEILING_NOTE, ATTRIBUTION, ""]
    return "\n".join(lines)


def comparison_table(results: list[BacktestResult]) -> pd.DataFrame:
    """One row per run, sorted by the metric that decides the question."""
    frame = pd.DataFrame([summarise(result) for result in results])
    if frame.empty:
        return frame
    return frame.sort_values("jpy_per_kwh_capacity_year", ascending=False, ignore_index=True)


def format_comparison(results: list[BacktestResult]) -> str:
    """Render the comparison as a fixed-width table."""
    frame = comparison_table(results)
    if frame.empty:
        return "No results to compare."

    header = (
        f"{'preset':<16} {'area':<9} {'tariff':<14} "
        f"{'JPY/kWh/yr':>11} {'annual JPY':>13} {'cyc/day':>8} {'active':>7}"
    )
    lines = [header, "-" * len(header)]
    for _, row in frame.iterrows():
        lines.append(
            f"{row['preset']:<16} {row['area']:<9} {row['tariff']:<14} "
            f"{row['jpy_per_kwh_capacity_year']:>11,.0f} "
            f"{row['annualised_jpy']:>13,.0f} "
            f"{row['cycles_per_day']:>8.2f} "
            f"{row['active_day_share']:>6.0%}"
        )
    span = f"{frame['date_from'].min()} .. {frame['date_to'].max()}"
    lines += ["", f"Window: {span}", CEILING_NOTE, ATTRIBUTION]
    return "\n".join(lines)


def _mean_buy_price(result: BacktestResult) -> float:
    """Buy price at the window's average market price, averaged over the day.

    Only used to give the breakeven-spread line a concrete anchor, so a
    representative price beats an exactly-weighted one.
    """
    market_mean = float(result.daily["market_mean"].mean())
    hours = slot_hour_of_day()
    return float(result.config.tariff.buy(np.full(SLOTS_PER_DAY, market_mean), hours).mean())
