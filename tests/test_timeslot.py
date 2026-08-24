import datetime as dt

import pandas as pd
import pytest

from jepx_arb.timeslot import JST, SLOTS_PER_DAY, slot_of, slot_start, slot_starts


def test_slot_one_is_midnight():
    assert slot_start(dt.date(2025, 4, 1), 1) == dt.datetime(2025, 4, 1, tzinfo=JST)


def test_last_slot_starts_at_2330():
    assert slot_start(dt.date(2025, 4, 1), SLOTS_PER_DAY).hour == 23
    assert slot_start(dt.date(2025, 4, 1), SLOTS_PER_DAY).minute == 30


@pytest.mark.parametrize("slot", [0, 49, -1])
def test_out_of_range_slot_rejected(slot):
    with pytest.raises(ValueError, match="slot must be"):
        slot_start(dt.date(2025, 4, 1), slot)


@pytest.mark.parametrize(
    ("time", "expected"),
    [((0, 0), 1), ((0, 29), 1), ((0, 30), 2), ((12, 0), 25), ((23, 59), 48)],
)
def test_slot_of_round_trips(time, expected):
    assert slot_of(dt.datetime(2025, 4, 1, *time)) == expected


def test_slot_starts_is_timezone_aware():
    result = slot_starts(pd.Series([dt.date(2025, 4, 1)]), pd.Series([25]))
    assert str(result.dt.tz) == "Asia/Tokyo"
    assert result.iloc[0].hour == 12
