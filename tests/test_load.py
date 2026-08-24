import datetime as dt

import pytest

from jepx_arb import load
from jepx_arb.timeslot import SLOTS_PER_DAY


@pytest.mark.parametrize("profile", ["household", "commercial", "flat"])
def test_profiles_have_one_value_per_slot(profile):
    assert len(load.for_date(profile, dt.date(2025, 4, 1), 100.0)) == SLOTS_PER_DAY


@pytest.mark.parametrize("profile", ["household", "commercial", "flat"])
def test_profiles_sum_to_the_requested_daily_energy(profile):
    values = load.for_date(profile, dt.date(2025, 4, 2), 100.0)  # a Wednesday
    assert values.sum() == pytest.approx(100.0)


def test_household_peaks_in_the_evening():
    """It has to, or the no-export case is being modelled wrong: the whole
    question is whether the load peak lines up with the price peak."""
    assert load.peak_hours(load.household(12.0))[0] >= 17.0


def test_commercial_sites_draw_less_at_weekends():
    saturday = load.for_date("commercial", dt.date(2025, 4, 5), 1000.0)
    wednesday = load.for_date("commercial", dt.date(2025, 4, 2), 1000.0)
    assert saturday.sum() < wednesday.sum()


def test_unknown_profile_lists_the_valid_ones():
    with pytest.raises(ValueError, match="household, commercial or flat"):
        load.for_date("nonsense", dt.date(2025, 4, 1), 10.0)


def test_negative_load_is_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        load.household(-5.0)
