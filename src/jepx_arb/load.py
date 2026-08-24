"""Synthetic site load profiles.

Needed only for the no-export case: a battery that cannot reverse-flow can
only discharge into whatever the site is drawing, so the load curve caps
the discharge and therefore caps the revenue.

These shapes are stand-ins with the right qualitative structure, not
measured data.  Feed in real half-hourly metering as soon as it exists --
:func:`from_series` takes it -- because the evening-peak overlap between
load and price is exactly what decides the answer, and a made-up curve
cannot settle it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from jepx_arb.timeslot import SLOTS_PER_DAY, slot_hour_of_day

# Relative shapes over the 48 slots; normalised to sum to 1 on use.
_HOUSEHOLD_SHAPE = np.array(
    [0.4] * 12          # 00:00-06:00 overnight base
    + [0.8] * 4         # 06:00-08:00 morning
    + [1.3] * 4         # 08:00-10:00 morning peak
    + [0.7] * 14        # 10:00-17:00 daytime, mostly out
    + [1.6] * 8         # 17:00-21:00 evening peak
    + [1.1] * 4         # 21:00-23:00 wind-down
    + [0.6] * 2         # 23:00-24:00
)

_COMMERCIAL_SHAPE = np.array(
    [0.3] * 14          # 00:00-07:00 closed
    + [0.9] * 4         # 07:00-09:00 opening
    + [1.5] * 18        # 09:00-18:00 trading hours
    + [1.0] * 6         # 18:00-21:00 closing
    + [0.4] * 6         # 21:00-24:00
)


def household(daily_kwh: float = 12.0) -> np.ndarray:
    """Residential day: overnight base, morning bump, hard evening peak."""
    return _scaled(_HOUSEHOLD_SHAPE, daily_kwh)


def commercial(daily_kwh: float = 1200.0, weekend_factor: float = 0.45) -> np.ndarray:
    """Weekday commercial/industrial day.  See :func:`for_date` for weekends."""
    del weekend_factor  # applied by for_date, kept here for signature symmetry
    return _scaled(_COMMERCIAL_SHAPE, daily_kwh)


def flat(daily_kwh: float) -> np.ndarray:
    """Constant draw -- useful when discharge should never be the binding limit."""
    return np.full(SLOTS_PER_DAY, daily_kwh / SLOTS_PER_DAY)


def for_date(
    profile: str, date: dt.date, daily_kwh: float, weekend_factor: float = 0.45
) -> np.ndarray:
    """Load for one calendar day, scaling commercial sites down at weekends."""
    if profile == "household":
        return household(daily_kwh)
    if profile == "commercial":
        scale = weekend_factor if date.weekday() >= 5 else 1.0
        return commercial(daily_kwh * scale)
    if profile == "flat":
        return flat(daily_kwh)
    raise ValueError(f"unknown load profile {profile!r}; use household, commercial or flat")


def from_series(series: pd.Series, date: dt.date) -> np.ndarray:
    """Pull one day out of a real half-hourly metering series.

    ``series`` must be indexed by ``(delivery_date, slot)`` with kWh values.
    """
    day = series.loc[date]
    values = np.zeros(SLOTS_PER_DAY)
    for slot, value in day.items():
        values[int(slot) - 1] = float(value)
    return values


def peak_hours(load: np.ndarray, top_n: int = 8) -> list[float]:
    """The hours where the load sits highest -- handy when sanity-checking
    whether a profile's peak lines up with the evening price peak."""
    order = np.argsort(load)[::-1][:top_n]
    return sorted(slot_hour_of_day()[order].tolist())


def _scaled(shape: np.ndarray, daily_kwh: float) -> np.ndarray:
    if len(shape) != SLOTS_PER_DAY:
        raise ValueError(f"shape must have {SLOTS_PER_DAY} slots, got {len(shape)}")
    if daily_kwh < 0:
        raise ValueError("daily_kwh must not be negative")
    return shape / shape.sum() * daily_kwh
