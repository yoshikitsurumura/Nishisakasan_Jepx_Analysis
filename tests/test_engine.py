import datetime as dt

import numpy as np
import pandas as pd
import pytest

from jepx_arb.backtest.engine import BacktestConfig, run_backtest
from jepx_arb.backtest.report import comparison_table, format_report, summarise
from jepx_arb.battery import BatterySpec
from jepx_arb.tariff import Tariff


def config(**kwargs) -> BacktestConfig:
    defaults = dict(
        spec=BatterySpec(
            name="t", capacity_kwh=10.0, power_kw=5.0,
            soc_min=0.0, soc_max=1.0, soc_initial=0.0,
            degradation_cost_jpy_per_kwh=1.0,
        ),
        tariff=Tariff.wholesale(buy_adder_jpy_kwh=0.0, sell_fee_jpy_kwh=0.0),
        area="tokyo",
    )
    return BacktestConfig(**{**defaults, **kwargs})


def test_each_day_is_solved_and_totalled(price_frame):
    result = run_backtest(price_frame, config())
    assert result.days == 3
    assert result.net_profit_jpy > 0
    assert result.net_profit_jpy == pytest.approx(result.daily["net_profit_jpy"].sum())


def test_identical_days_produce_identical_profits(price_frame):
    """The fixture repeats one duck curve three times; with the SOC pinned
    back to its start each day, the three days must be interchangeable."""
    daily = run_backtest(price_frame, config()).daily
    assert daily["net_profit_jpy"].nunique() == 1


def test_continuous_horizon_is_never_worse_than_daily(price_frame):
    daily = run_backtest(price_frame, config(horizon="day"))
    continuous = run_backtest(price_frame, config(horizon="continuous"))
    assert continuous.net_profit_jpy >= daily.net_profit_jpy - 1e-6


def test_incomplete_days_are_skipped_not_guessed_at(price_frame):
    truncated = price_frame[
        ~((price_frame["delivery_date"] == dt.date(2025, 4, 2)) & (price_frame["slot"] > 40))
    ]
    result = run_backtest(truncated, config())
    assert result.days == 2
    assert [date for date, _ in result.skipped_days] == [dt.date(2025, 4, 2)]


def test_no_export_battery_demands_a_load_profile():
    with pytest.raises(ValueError, match="capped by site load"):
        config(spec=BatterySpec(name="t", capacity_kwh=10.0, power_kw=5.0, allow_export=False))


def test_load_profile_caps_what_a_no_export_battery_earns(price_frame):
    exporting = config()
    captive = config(
        spec=exporting.spec.with_(allow_export=False),
        load_profile="household",
        load_daily_kwh=2.0,
    )
    assert run_backtest(price_frame, captive).net_profit_jpy < (
        run_backtest(price_frame, exporting).net_profit_jpy
    )


def test_unknown_area_says_what_is_available(price_frame):
    with pytest.raises(ValueError, match="no rows for area"):
        run_backtest(price_frame, config(area="hokkaido"))


def test_missing_columns_are_reported(price_frame):
    with pytest.raises(ValueError, match="missing column"):
        run_backtest(price_frame.drop(columns=["price_jpy_kwh"]), config())


def test_schedule_is_opt_in(price_frame):
    assert run_backtest(price_frame, config()).schedule is None
    schedule = run_backtest(price_frame, config(keep_schedule=True)).schedule
    assert len(schedule) == 3 * 48
    assert {"charge_kwh", "discharge_kwh", "soc_kwh"} <= set(schedule.columns)


def test_annualisation_scales_a_short_window(price_frame):
    result = run_backtest(price_frame, config())
    assert result.annualised_profit_jpy == pytest.approx(result.net_profit_jpy / 3 * 365)
    assert result.profit_per_kwh_capacity_year == pytest.approx(
        result.annualised_profit_jpy / 10.0
    )


def test_bad_horizon_is_rejected():
    with pytest.raises(ValueError, match="horizon must be"):
        config(horizon="weekly")


def test_report_renders_and_carries_the_caveat(price_frame):
    text = format_report(run_backtest(price_frame, config()))
    assert "Perfect foresight" in text
    assert "JEPX" in text


def test_empty_result_reports_cleanly():
    empty = pd.DataFrame(columns=["delivery_date", "slot", "price_jpy_kwh"])
    result = run_backtest(empty, config())
    assert result.days == 0
    assert "nothing to report" in format_report(result)


def test_comparison_ranks_by_profit_per_kwh(price_frame):
    strong = run_backtest(price_frame, config())
    weak = run_backtest(price_frame, config(spec=strong.config.spec.with_(
        degradation_cost_jpy_per_kwh=8.0)))
    table = comparison_table([weak, strong])
    assert table.iloc[0]["jpy_per_kwh_capacity_year"] >= table.iloc[1][
        "jpy_per_kwh_capacity_year"
    ]


def test_summary_is_json_friendly(price_frame):
    stats = summarise(run_backtest(price_frame, config()))
    assert stats["days"] == 3
    assert all(not isinstance(value, np.ndarray) for value in stats.values())
