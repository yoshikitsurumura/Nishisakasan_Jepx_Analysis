import json
from dataclasses import replace

import pytest

from jepx_arb.sources.parser import parse_spot_summary
from jepx_arb.sources.store import SpotStore


@pytest.fixture
def frame(sample_csv_bytes):
    return parse_spot_summary(sample_csv_bytes, source="fixture")


@pytest.fixture
def store(tmp_path):
    return SpotStore(tmp_path / "spot")


def test_first_write_counts_everything_as_new(store, frame, sample_csv_bytes):
    result = store.write(frame, 2024, sample_csv_bytes)
    assert result.rows_added == len(frame.prices)
    assert result.rows_changed == 0
    assert result.changed


def test_rewriting_the_same_file_changes_nothing(store, frame, sample_csv_bytes):
    """The yearly file is re-downloaded whole every day, so the common case
    is a fetch that should be a no-op."""
    store.write(frame, 2024, sample_csv_bytes)
    result = store.write(frame, 2024, sample_csv_bytes)
    assert (result.rows_added, result.rows_changed) == (0, 0)
    assert not result.changed
    assert "no change" in result.describe()


def test_republished_prices_overwrite_and_are_counted(store, frame, sample_csv_bytes):
    """JEPX does correct past days; the store must take the new value and
    say how many rows moved."""
    store.write(frame, 2024, sample_csv_bytes)
    revised = frame.prices.copy()
    target = (revised["slot"] == 20) & (revised["area"] == "tokyo")
    revised.loc[target, "price_jpy_kwh"] = 99.99

    result = store.write(replace(frame, prices=revised), 2024, sample_csv_bytes)
    assert result.rows_added == 0
    assert result.rows_changed == int(target.sum())

    stored = store.load(areas=["tokyo"])
    assert (stored[stored["slot"] == 20]["price_jpy_kwh"] == 99.99).all()


def test_new_days_are_appended(store, frame, sample_csv_bytes):
    partial = frame.prices[frame.prices["delivery_date"] < frame.prices["delivery_date"].max()]
    store.write(replace(frame, prices=partial), 2024, sample_csv_bytes)
    result = store.write(frame, 2024, sample_csv_bytes)
    assert result.rows_added == len(frame.prices) - len(partial)


def test_load_filters_by_date_and_area(store, frame, sample_csv_bytes):
    store.write(frame, 2024, sample_csv_bytes)
    subset = store.load(start="2024-04-02", end="2024-04-02", areas=["tokyo", "kansai"])
    assert set(subset["area"]) == {"tokyo", "kansai"}
    assert subset["delivery_date"].nunique() == 1
    assert len(subset) == 2 * 48


def test_load_wide_gives_one_column_per_area(store, frame, sample_csv_bytes):
    store.write(frame, 2024, sample_csv_bytes)
    wide = store.load_wide(areas=["tokyo", "kansai"])
    assert list(wide.columns) == ["kansai", "tokyo"]
    assert len(wide) == 3 * 48


def test_loading_an_empty_store_says_what_to_run(store):
    with pytest.raises(FileNotFoundError, match="jepx-arb fetch"):
        store.load()


def test_manifest_records_the_download_fingerprint(store, frame, sample_csv_bytes):
    store.write(frame, 2024, sample_csv_bytes)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))["2024"]
    assert manifest["date_min"] == "2024-04-01"
    assert manifest["date_max"] == "2024-04-03"
    assert len(manifest["raw_sha256"]) == 64


def test_years_lists_what_is_on_disk(store, frame, sample_csv_bytes):
    store.write(frame, 2024, sample_csv_bytes)
    store.write(frame, 2025, sample_csv_bytes)
    assert store.years() == [2024, 2025]


def test_round_trip_preserves_the_timezone(store, frame, sample_csv_bytes):
    store.write(frame, 2024, sample_csv_bytes)
    assert str(store.load()["ts_jst"].dt.tz) == "Asia/Tokyo"
