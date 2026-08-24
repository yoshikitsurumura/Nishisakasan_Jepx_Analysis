"""JEPX time-slot (コマ) handling.

JEPX divides every delivery day into 48 half-hour slots numbered 1..48.
Slot 1 covers 00:00-00:30 JST, slot 48 covers 23:30-24:00 JST.

Japan has no daylight saving time, so a delivery day is always exactly
48 slots and the mapping below is unconditional.  Everything in this
project keeps ``delivery_date`` + ``slot`` as the canonical key and
carries ``ts_jst`` (tz-aware ``Asia/Tokyo``, slot *start*) as a
convenience column.  The timezone is attached deliberately: joining
JEPX against UTC-stamped weather data with naive timestamps is a silent
nine-hour bug.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

JST = ZoneInfo("Asia/Tokyo")

SLOTS_PER_DAY = 48
SLOT_MINUTES = 30
SLOT_HOURS = SLOT_MINUTES / 60.0


def slot_start(delivery_date: dt.date, slot: int) -> dt.datetime:
    """Return the tz-aware JST start time of ``slot`` on ``delivery_date``."""
    if not 1 <= slot <= SLOTS_PER_DAY:
        raise ValueError(f"slot must be 1..{SLOTS_PER_DAY}, got {slot}")
    midnight = dt.datetime(
        delivery_date.year, delivery_date.month, delivery_date.day, tzinfo=JST
    )
    return midnight + dt.timedelta(minutes=SLOT_MINUTES * (slot - 1))


def slot_starts(dates: pd.Series, slots: pd.Series) -> pd.Series:
    """Vectorised :func:`slot_start` for whole columns."""
    base = pd.to_datetime(dates).dt.tz_localize(JST)
    offset = pd.to_timedelta((slots.astype("int64") - 1) * SLOT_MINUTES, unit="m")
    return base + offset


def slot_of(ts: dt.datetime) -> int:
    """Return the 1-based slot number containing ``ts`` (interpreted in JST)."""
    local = ts.astimezone(JST) if ts.tzinfo else ts.replace(tzinfo=JST)
    return local.hour * 2 + (1 if local.minute < SLOT_MINUTES else 2)


def slot_hour_of_day() -> np.ndarray:
    """Fractional hour-of-day at the start of each slot, shape ``(48,)``."""
    return np.arange(SLOTS_PER_DAY) * SLOT_HOURS
