import pytest

from jepx_arb.battery import BatterySpec


def spec(**kwargs) -> BatterySpec:
    defaults = dict(name="t", capacity_kwh=10.0, power_kw=5.0)
    return BatterySpec(**{**defaults, **kwargs})


def test_usable_window_excludes_the_reserved_ends():
    assert spec(soc_min=0.1, soc_max=0.9, soc_initial=0.1).usable_kwh == pytest.approx(8.0)


def test_round_trip_is_the_product_of_both_directions():
    assert spec(charge_efficiency=0.9, discharge_efficiency=0.9).round_trip_efficiency == (
        pytest.approx(0.81)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity_kwh": 0},
        {"power_kw": -1},
        {"charge_efficiency": 1.5},
        {"soc_min": 0.9, "soc_max": 0.5},
        {"soc_initial": 0.99, "soc_max": 0.9},
    ],
)
def test_invalid_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        spec(**kwargs)


def test_breakeven_spread_scales_the_efficiency_loss_with_the_buy_price():
    battery = spec(charge_efficiency=0.9, discharge_efficiency=0.9,
                   degradation_cost_jpy_per_kwh=3.0)
    # Wear is fixed; the efficiency penalty is proportional to what energy costs.
    cheap = battery.breakeven_spread_jpy_kwh(10.0)
    dear = battery.breakeven_spread_jpy_kwh(20.0)
    assert cheap == pytest.approx(3.0 + 10.0 * (1 / 0.81 - 1))
    assert dear - cheap == pytest.approx(10.0 * (1 / 0.81 - 1))


def test_capex_spreads_over_lifetime_ac_output():
    cost = BatterySpec.cost_per_ac_kwh(
        capex_jpy=1_000_000, cycle_life=1_000, capacity_kwh=10,
        depth_of_discharge=1.0, discharge_efficiency=1.0,
    )
    assert cost == pytest.approx(100.0)  # 1M JPY over 10,000 kWh delivered


def test_from_capex_uses_the_specs_own_depth_of_discharge():
    battery = BatterySpec.from_capex(
        "x", capacity_kwh=10, power_kw=5, capex_jpy=100_000, cycle_life=1_000,
        soc_min=0.0, soc_max=1.0, discharge_efficiency=1.0,
    )
    assert battery.degradation_cost_jpy_per_kwh == pytest.approx(10.0)


def test_with_returns_a_copy_and_revalidates():
    original = spec()
    assert original.with_(capacity_kwh=20).capacity_kwh == 20
    assert original.capacity_kwh == 10
    with pytest.raises(ValueError):
        original.with_(capacity_kwh=-5)
