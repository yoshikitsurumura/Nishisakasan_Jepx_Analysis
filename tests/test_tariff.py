import numpy as np
import pytest

from jepx_arb.tariff import Tariff
from jepx_arb.timeslot import slot_hour_of_day


@pytest.fixture
def hours():
    return slot_hour_of_day()


@pytest.fixture
def market():
    return np.full(48, 10.0)


def test_wholesale_pays_a_bit_more_and_earns_a_bit_less(market, hours):
    buy, sell = Tariff.wholesale(buy_adder_jpy_kwh=1.0, sell_fee_jpy_kwh=0.5).prices(market, hours)
    assert buy[0] == pytest.approx(11.0)
    assert sell[0] == pytest.approx(9.5)


def test_self_consumption_values_discharge_at_the_avoided_purchase(market, hours):
    tariff = Tariff.market_linked_retail(adder_jpy_kwh=12.0, can_export=False)
    buy, sell = tariff.prices(market, hours)
    assert np.allclose(buy, sell)
    assert buy[0] == pytest.approx(22.0)


def test_self_consumption_spread_beats_wholesale(hours):
    """The point of the no-export case: the adder widens the spread rather
    than narrowing it, because it applies on both sides."""
    market = np.array([1.0] * 24 + [30.0] * 24)
    retail_buy, retail_sell = Tariff.market_linked_retail(adder_jpy_kwh=12.0).prices(market, hours)
    wholesale_buy, wholesale_sell = Tariff.wholesale().prices(market, hours)
    assert (retail_sell.max() - retail_buy.min()) > (wholesale_sell.max() - wholesale_buy.min())


def test_export_at_a_fixed_buyback_price(market, hours):
    tariff = Tariff.market_linked_retail(can_export=True, export_price_jpy_kwh=8.0)
    _, sell = tariff.prices(market, hours)
    assert np.allclose(sell, 8.0)


def test_flat_tariff_offers_no_spread_at_all(market, hours):
    buy, sell = Tariff.flat(price_jpy_kwh=31.0).prices(market, hours)
    assert np.allclose(buy, 31.0) and np.allclose(sell, 31.0)


def test_time_of_use_ignores_the_market(hours):
    tariff = Tariff.time_of_use()
    cheap, _ = tariff.prices(np.full(48, 5.0), hours)
    dear, _ = tariff.prices(np.full(48, 50.0), hours)
    assert np.allclose(cheap, dear)


def test_time_of_use_applies_the_night_rate_overnight(hours):
    buy = Tariff.time_of_use().buy(np.zeros(48), hours)
    assert buy[0] == pytest.approx(21.0)     # 00:00
    assert buy[20] == pytest.approx(38.0)    # 10:00
    assert buy[47] == pytest.approx(21.0)    # 23:30


def test_uncovered_hours_in_a_schedule_are_rejected(hours):
    tariff = Tariff.time_of_use(schedule=((0.0, 12.0, 20.0),))
    with pytest.raises(ValueError, match="leaves hours uncovered"):
        tariff.buy(np.zeros(48), hours)
