"""The seam between "where prices come from" and everything else.

JEPX publishes its site under a disclaimer that reserves the right to
change or withdraw the content without notice, so treat any scraper as a
component with a limited lifetime.  Keeping the interface narrow means a
switch to a paid feed later is a new file in this package, not a rewrite.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

# Canonical area keys.  "system" is the nationwide system price
# (システムプライス), the other nine are the area prices (エリアプライス).
AREAS: tuple[str, ...] = (
    "system",
    "hokkaido",
    "tohoku",
    "tokyo",
    "chubu",
    "hokuriku",
    "kansai",
    "chugoku",
    "shikoku",
    "kyushu",
)

DELIVERY_AREAS: tuple[str, ...] = AREAS[1:]

# Japanese label -> canonical key, used by the CSV parser.
AREA_LABELS_JA: dict[str, str] = {
    "北海道": "hokkaido",
    "東北": "tohoku",
    "東京": "tokyo",
    "中部": "chubu",
    "北陸": "hokuriku",
    "関西": "kansai",
    "中国": "chugoku",
    "四国": "shikoku",
    "九州": "kyushu",
}

PRICE_COLUMNS: tuple[str, ...] = ("delivery_date", "slot", "ts_jst", "area", "price_jpy_kwh")
VOLUME_COLUMNS: tuple[str, ...] = (
    "delivery_date",
    "slot",
    "ts_jst",
    "sell_bid_kwh",
    "buy_bid_kwh",
    "contract_kwh",
)


@dataclass(frozen=True)
class SpotFrame:
    """One parsed spot-market file.

    ``prices`` is long-format (one row per date/slot/area) because that is
    what stores and joins cleanly; use :meth:`wide` when a backtest wants
    one column per area.  ``volumes`` is market-wide and therefore separate
    rather than duplicated across ten area rows.
    """

    prices: pd.DataFrame
    volumes: pd.DataFrame
    source: str = ""

    def wide(self) -> pd.DataFrame:
        """Pivot prices to one column per area, indexed by date and slot."""
        return (
            self.prices.pivot_table(
                index=["delivery_date", "slot"], columns="area", values="price_jpy_kwh"
            )
            .rename_axis(columns=None)
            .sort_index()
        )

    @property
    def date_range(self) -> tuple[dt.date, dt.date] | tuple[None, None]:
        if self.prices.empty:
            return (None, None)
        return (self.prices["delivery_date"].min(), self.prices["delivery_date"].max())

    def __len__(self) -> int:
        return len(self.prices)


@runtime_checkable
class PriceSource(Protocol):
    """Anything that can hand back spot prices for a year."""

    name: str

    def fetch_year(self, year: int) -> SpotFrame:
        """Return every settled slot published for the given JEPX file year."""
        ...
