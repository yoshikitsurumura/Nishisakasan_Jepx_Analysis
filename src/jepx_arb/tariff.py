"""Turning a market price into the two prices a battery actually faces.

A JEPX area price is not what anyone pays.  What matters to the dispatch
is the *buy* price of a kWh pulled from the grid and the *sell* value of a
kWh pushed back, and the gap between those two is what decides whether a
segment is a business:

* 系統用 (grid-scale, direct market participant) sees close to the raw
  market price on both sides, minus fees.  Both directions move with JEPX.
* 市場連動型小売 (market-linked retail) sees market + a fixed stack of
  wheeling charges, levies and retail margin on the buy side.  If the site
  cannot export, the "sell" side is the purchase it avoids -- the same
  inflated price -- which is *better*, not worse, than wholesale.
* 従量・時間帯別 (flat or time-of-use retail) does not move with JEPX at
  all.  Backtesting it is still worth doing: it is the baseline the
  market-linked case has to beat.

Every adder below is a **placeholder**, not a quote.  Wheeling charges
differ by TSO and voltage class, the renewable levy changes annually, and
retail margins are negotiated.  Replace them with the contract's real
numbers before showing anyone a profit figure.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# (start_hour, end_hour, jpy_per_kwh); end_hour of 24 means midnight.
TouSchedule = tuple[tuple[float, float, float], ...]

# A common Japanese residential night-discount shape, rounded.
NIGHT_DISCOUNT_TOU: TouSchedule = (
    (0.0, 7.0, 21.0),
    (7.0, 23.0, 38.0),
    (23.0, 24.0, 21.0),
)


@dataclass(frozen=True)
class Tariff:
    """Maps market prices to the buy and sell prices seen at the meter."""

    name: str
    buy_scale: float = 1.0
    """How much of the market price passes through to the buy side.
    1.0 = fully market-linked, 0.0 = a fixed retail tariff."""

    buy_adder_jpy_kwh: float = 0.0
    """Wheeling + levy + margin stacked on top of the passed-through price."""

    sell_scale: float = 1.0
    sell_fee_jpy_kwh: float = 0.0

    sell_equals_buy: bool = False
    """Self-consumption: a discharged kWh is a kWh not bought, so it is
    worth the buy price rather than the wholesale price."""

    tou_jpy_kwh: TouSchedule | None = None
    """Time-of-use component added to the buy side, by hour of day."""

    def buy(self, market: np.ndarray, hours: np.ndarray) -> np.ndarray:
        """Buy price per kWh for each slot."""
        prices = np.asarray(market, dtype=float) * self.buy_scale + self.buy_adder_jpy_kwh
        if self.tou_jpy_kwh:
            prices = prices + _tou_values(self.tou_jpy_kwh, hours)
        return prices

    def sell(self, market: np.ndarray, hours: np.ndarray) -> np.ndarray:
        """Value per kWh discharged, for each slot."""
        if self.sell_equals_buy:
            return self.buy(market, hours)
        return np.asarray(market, dtype=float) * self.sell_scale - self.sell_fee_jpy_kwh

    def prices(self, market: np.ndarray, hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.buy(market, hours), self.sell(market, hours)

    def with_(self, **changes) -> Tariff:
        return replace(self, **changes)

    # -- constructors for the three segments that actually differ ---------

    @classmethod
    def wholesale(
        cls,
        name: str = "wholesale",
        buy_adder_jpy_kwh: float = 1.0,
        sell_fee_jpy_kwh: float = 0.5,
    ) -> Tariff:
        """Grid-scale battery trading directly in the spot market.

        The adders stand in for transmission loss adjustment, the JEPX
        transaction fee and the generator-side wheeling charge.
        """
        return cls(
            name=name,
            buy_scale=1.0,
            buy_adder_jpy_kwh=buy_adder_jpy_kwh,
            sell_scale=1.0,
            sell_fee_jpy_kwh=sell_fee_jpy_kwh,
        )

    @classmethod
    def market_linked_retail(
        cls,
        name: str = "market-linked",
        adder_jpy_kwh: float = 12.0,
        can_export: bool = False,
        export_price_jpy_kwh: float | None = None,
    ) -> Tariff:
        """Retail plan whose energy charge tracks JEPX, plus a fixed stack.

        With ``can_export=False`` (most residential systems) the sell side
        is the avoided purchase.  With ``can_export=True`` and an
        ``export_price_jpy_kwh``, exports earn that fixed price instead --
        a FIT/FIP-style buyback rather than the market.
        """
        if not can_export:
            return cls(
                name=name,
                buy_scale=1.0,
                buy_adder_jpy_kwh=adder_jpy_kwh,
                sell_equals_buy=True,
            )
        if export_price_jpy_kwh is None:
            return cls(name=name, buy_scale=1.0, buy_adder_jpy_kwh=adder_jpy_kwh)
        return cls(
            name=name,
            buy_scale=1.0,
            buy_adder_jpy_kwh=adder_jpy_kwh,
            sell_scale=0.0,
            sell_fee_jpy_kwh=-export_price_jpy_kwh,
        )

    @classmethod
    def time_of_use(
        cls,
        name: str = "time-of-use",
        schedule: TouSchedule = NIGHT_DISCOUNT_TOU,
    ) -> Tariff:
        """Fixed time-of-use retail tariff, self-consumption only.

        Independent of JEPX by construction -- included as the baseline a
        market-linked plan has to beat.
        """
        return cls(name=name, buy_scale=0.0, tou_jpy_kwh=schedule, sell_equals_buy=True)

    @classmethod
    def flat(cls, name: str = "flat", price_jpy_kwh: float = 31.0) -> Tariff:
        """Single-rate retail tariff.  Arbitrage against it is impossible;
        a run against this should come out negative, and that is the point."""
        return cls(
            name=name, buy_scale=0.0, buy_adder_jpy_kwh=price_jpy_kwh, sell_equals_buy=True
        )


def _tou_values(schedule: TouSchedule, hours: np.ndarray) -> np.ndarray:
    """Look up the time-of-use rate for each slot's start hour."""
    hours = np.asarray(hours, dtype=float)
    values = np.full(hours.shape, np.nan)
    for start, end, price in schedule:
        window = (hours >= start) & (hours < end)
        values[window] = price
    if np.isnan(values).any():
        uncovered = sorted(set(hours[np.isnan(values)].tolist()))
        raise ValueError(f"time-of-use schedule leaves hours uncovered: {uncovered}")
    return values
