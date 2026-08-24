from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jepx_arb.timeslot import SLOTS_PER_DAY

FIXTURE = Path(__file__).parent / "fixtures" / "spot_summary_sample.csv"


@pytest.fixture
def sample_csv_bytes() -> bytes:
    return FIXTURE.read_bytes()


@pytest.fixture
def duck_curve() -> np.ndarray:
    """A day with a cheap solar middle and an expensive evening peak."""
    prices = np.full(SLOTS_PER_DAY, 12.0)
    prices[20:28] = 1.0      # 10:00-14:00
    prices[34:42] = 30.0     # 17:00-21:00
    return prices


@pytest.fixture
def price_frame(duck_curve) -> pd.DataFrame:
    """Three identical duck-curve days in long format for one area."""
    rows = []
    for offset in range(3):
        date = dt.date(2025, 4, 1) + dt.timedelta(days=offset)
        for slot in range(1, SLOTS_PER_DAY + 1):
            rows.append(
                {
                    "delivery_date": date,
                    "slot": slot,
                    "area": "tokyo",
                    "price_jpy_kwh": float(duck_curve[slot - 1]),
                }
            )
    return pd.DataFrame(rows)
