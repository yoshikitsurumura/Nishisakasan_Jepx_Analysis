import pytest

from jepx_arb.sources.parser import (
    SpotCsvFormatError,
    normalise_header,
    parse_spot_summary,
    slot_count_anomalies,
)


def test_parses_shift_jis_fixture(sample_csv_bytes):
    frame = parse_spot_summary(sample_csv_bytes, source="fixture")
    # 3 days x 48 slots x 10 series, less the one blanked cell in the fixture.
    assert len(frame.prices) == 3 * 48 * 10 - 1
    assert set(frame.prices["area"]) == {
        "system", "hokkaido", "tohoku", "tokyo", "chubu",
        "hokuriku", "kansai", "chugoku", "shikoku", "kyushu",
    }


def test_timestamps_carry_jst(sample_csv_bytes):
    frame = parse_spot_summary(sample_csv_bytes)
    assert str(frame.prices["ts_jst"].dt.tz) == "Asia/Tokyo"


def test_volumes_are_market_wide_not_per_area(sample_csv_bytes):
    frame = parse_spot_summary(sample_csv_bytes)
    assert len(frame.volumes) == 3 * 48
    assert frame.volumes["contract_kwh"].notna().all()


def test_blank_price_becomes_a_missing_row_not_a_zero(sample_csv_bytes):
    anomalies = slot_count_anomalies(parse_spot_summary(sample_csv_bytes).prices)
    assert len(anomalies) == 1
    assert anomalies.iloc[0]["slots"] == 47


def test_unknown_trailing_columns_are_ignored(sample_csv_bytes):
    """The fixture carries alpha and avoidable-cost columns; they must not
    leak in as areas."""
    frame = parse_spot_summary(sample_csv_bytes)
    assert frame.prices["area"].nunique() == 10


def test_missing_required_column_is_fatal_and_says_what_it_saw():
    csv = "受渡日,売り入札量(kWh)\n2025/04/01,100\n"
    with pytest.raises(SpotCsvFormatError) as error:
        parse_spot_summary(csv.encode("cp932"), source="broken.csv")
    message = str(error.value)
    assert "slot code" in message and "system price" in message
    assert "broken.csv" in message


def test_slot_outside_range_is_fatal():
    csv = "受渡日,時刻コード,システムプライス(円/kWh)\n2025/04/01,49,10.0\n"
    with pytest.raises(SpotCsvFormatError, match="outside 1..48"):
        parse_spot_summary(csv.encode("cp932"))


def test_unparseable_date_is_fatal():
    csv = "受渡日,時刻コード,システムプライス(円/kWh)\nnot-a-date,1,10.0\n"
    with pytest.raises(SpotCsvFormatError, match="unparseable delivery dates"):
        parse_spot_summary(csv.encode("cp932"))


def test_utf8_export_also_parses():
    csv = "受渡日,時刻コード,システムプライス(円/kWh)\n2025/04/01,1,10.5\n"
    frame = parse_spot_summary(csv.encode("utf-8"))
    assert frame.prices.iloc[0]["price_jpy_kwh"] == 10.5


def test_thousands_separators_are_stripped():
    csv = (
        "受渡日,時刻コード,システムプライス(円/kWh),売り入札量(kWh)\n"
        '2025/04/01,1,10.5,"1,234,567"\n'
    )
    frame = parse_spot_summary(csv.encode("cp932"))
    assert frame.volumes.iloc[0]["sell_bid_kwh"] == 1234567


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("システムプライス(円/kWh)", "システムプライス"),
        ("エリアプライス東京（円/kWh）", "エリアプライス東京"),
        (" 時刻コード ", "時刻コード"),
    ],
)
def test_header_normalisation(raw, expected):
    assert normalise_header(raw) == expected
