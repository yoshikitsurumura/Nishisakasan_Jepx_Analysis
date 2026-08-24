"""The dispatch LP is checked against arithmetic done by hand.

Every expected value below is derivable on paper, which is the only way to
be sure the solver is answering the question we think it is.
"""

import numpy as np
import pytest

from jepx_arb.backtest.dispatch import optimise
from jepx_arb.battery import BatterySpec


@pytest.fixture
def ideal() -> BatterySpec:
    """Lossless, wear-free, and never power-limited: profit is pure arbitrage."""
    return BatterySpec(
        name="ideal", capacity_kwh=10.0, power_kw=20.0,
        charge_efficiency=1.0, discharge_efficiency=1.0,
        soc_min=0.0, soc_max=1.0, soc_initial=0.0,
    )


CHEAP_THEN_DEAR = np.array([1.0, 1.0, 10.0, 10.0])


def test_buys_low_sells_high(ideal):
    result = optimise(ideal, CHEAP_THEN_DEAR, CHEAP_THEN_DEAR)
    # Fill 10 kWh at 1 JPY, empty it at 10 JPY: 100 - 10 = 90.
    assert result.net_profit_jpy == pytest.approx(90.0)
    assert result.charge_kwh[:2].sum() == pytest.approx(10.0)
    assert result.discharge_kwh[2:].sum() == pytest.approx(10.0)


def test_efficiency_makes_you_buy_more_than_you_sell(ideal):
    lossy = ideal.with_(charge_efficiency=0.95, discharge_efficiency=0.95)
    result = optimise(lossy, CHEAP_THEN_DEAR, CHEAP_THEN_DEAR)
    assert result.charged_kwh > result.discharged_kwh
    assert result.discharged_kwh == pytest.approx(result.charged_kwh * 0.9025, rel=1e-6)


def test_profit_identity_holds(ideal):
    lossy = ideal.with_(charge_efficiency=0.95, discharge_efficiency=0.95,
                        degradation_cost_jpy_per_kwh=2.0)
    result = optimise(lossy, CHEAP_THEN_DEAR, CHEAP_THEN_DEAR)
    assert result.net_profit_jpy == pytest.approx(
        result.revenue_jpy - result.energy_cost_jpy - result.degradation_jpy
    )


def test_stays_still_when_the_spread_cannot_cover_the_wear(ideal):
    battery = ideal.with_(charge_efficiency=0.95, discharge_efficiency=0.95,
                          degradation_cost_jpy_per_kwh=5.0)
    prices = np.array([9.0, 9.0, 10.0, 10.0])
    assert battery.breakeven_spread_jpy_kwh(9.0) > 1.0
    result = optimise(battery, prices, prices)
    assert result.discharged_kwh == pytest.approx(0.0, abs=1e-6)


def test_wears_through_a_spread_that_does_cover_it(ideal):
    battery = ideal.with_(degradation_cost_jpy_per_kwh=5.0)
    result = optimise(battery, CHEAP_THEN_DEAR, CHEAP_THEN_DEAR)
    assert result.discharged_kwh == pytest.approx(10.0)
    assert result.net_profit_jpy == pytest.approx(100.0 - 10.0 - 50.0)


def test_discharge_is_capped_by_site_load(ideal):
    """The no-export case: you can only discharge into what the site draws."""
    result = optimise(ideal, CHEAP_THEN_DEAR, CHEAP_THEN_DEAR,
                      max_discharge_kwh=np.array([0.0, 0.0, 3.0, 3.0]))
    assert result.discharged_kwh == pytest.approx(6.0)
    assert result.net_profit_jpy == pytest.approx(60.0 - 6.0)


def test_power_rating_limits_energy_per_slot():
    battery = BatterySpec(
        name="slow", capacity_kwh=10.0, power_kw=2.0,
        charge_efficiency=1.0, discharge_efficiency=1.0,
        soc_min=0.0, soc_max=1.0, soc_initial=0.0,
    )
    result = optimise(battery, CHEAP_THEN_DEAR, CHEAP_THEN_DEAR)
    assert result.charge_kwh.max() <= 1.0 + 1e-9  # 2 kW over a half-hour slot


def test_terminal_initial_forbids_finishing_empty(ideal):
    half_full = ideal.with_(soc_initial=0.5)
    flat = np.full(4, 10.0)
    assert optimise(half_full, flat, flat, terminal="initial").net_profit_jpy == (
        pytest.approx(0.0, abs=1e-6)
    )
    # Left free, the opening 5 kWh is simply sold: 5 * 10 = 50.
    assert optimise(half_full, flat, flat, terminal="free").net_profit_jpy == (
        pytest.approx(50.0)
    )


def test_cycle_cap_limits_throughput(ideal):
    result = optimise(ideal.with_(max_cycles_per_day=0.5), CHEAP_THEN_DEAR, CHEAP_THEN_DEAR)
    assert result.discharged_kwh == pytest.approx(5.0)


def test_soc_stays_inside_the_window(ideal):
    battery = ideal.with_(soc_min=0.2, soc_max=0.8, soc_initial=0.2)
    result = optimise(battery, CHEAP_THEN_DEAR, CHEAP_THEN_DEAR)
    assert result.soc_kwh.min() >= battery.soc_min_kwh - 1e-6
    assert result.soc_kwh.max() <= battery.soc_max_kwh + 1e-6


def test_soc_start_outside_the_window_is_rejected(ideal):
    with pytest.raises(ValueError, match="outside the usable window"):
        optimise(ideal.with_(soc_min=0.2, soc_max=0.8, soc_initial=0.2),
                 CHEAP_THEN_DEAR, CHEAP_THEN_DEAR, soc_start_kwh=9.0)


def test_doing_nothing_is_always_available(ideal):
    """Standing still satisfies every constraint, so the LP cannot be
    infeasible for any valid starting SOC.  Worth pinning: it means a bad
    day produces a zero, never a crashed nightly run."""
    battery = ideal.with_(power_kw=2.0, soc_initial=0.5, max_cycles_per_day=0.0)
    result = optimise(battery, np.array([1.0]), np.array([10.0]), terminal="initial")
    assert result.discharged_kwh == pytest.approx(0.0, abs=1e-9)
    assert result.net_profit_jpy == pytest.approx(0.0, abs=1e-9)


def test_empty_horizon_is_rejected(ideal):
    with pytest.raises(ValueError, match="empty horizon"):
        optimise(ideal, np.array([]), np.array([]))


def test_unknown_terminal_mode_is_rejected(ideal):
    with pytest.raises(ValueError, match="terminal must be"):
        optimise(ideal, CHEAP_THEN_DEAR, CHEAP_THEN_DEAR, terminal="whenever")


def test_mismatched_price_arrays_are_rejected(ideal):
    with pytest.raises(ValueError, match="same length"):
        optimise(ideal, np.array([1.0, 2.0]), np.array([1.0]))


def test_same_slot_round_trip_arbitrage_is_warned_about(ideal, caplog):
    """A tariff paying more to sell than to buy is a configuration error,
    and the LP will happily exploit it, so it must be flagged."""
    with caplog.at_level("WARNING"):
        optimise(ideal, np.full(4, 1.0), np.full(4, 5.0))
    assert "same-slot round trip" in caplog.text
