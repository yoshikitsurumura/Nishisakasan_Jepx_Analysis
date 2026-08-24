"""Tolerant parser for JEPX ``spot_summary_YYYY.csv``.

The file is Shift_JIS with a Japanese header row whose column *set* has
changed several times over the years (avoidable-cost columns, alpha
columns and FIP reference prices have come and gone).  So nothing here
addresses a column by position: every column is located by matching its
header, and anything unrecognised is ignored rather than fatal.

The flip side is that a column we actually need must never be guessed at.
If the date, slot or system-price column cannot be found the parser
raises :class:`SpotCsvFormatError` listing what it did see, so a failed
nightly run says what changed instead of quietly writing rubbish.
"""

from __future__ import annotations

import re
import unicodedata
from io import StringIO

import pandas as pd

from jepx_arb.sources.base import AREA_LABELS_JA, SpotFrame
from jepx_arb.timeslot import SLOTS_PER_DAY, slot_starts

ENCODINGS: tuple[str, ...] = ("cp932", "utf-8-sig", "utf-8")

# Headers carry unit suffixes like "(円/kWh)" and occasionally full-width
# parentheses or stray spaces; strip all of that before matching.
_DATE_PAT = re.compile(r"(受渡日|年月日|受渡年月日|日付)")
_SLOT_PAT = re.compile(r"(時刻コード|コマ|時間コード)")
_SYSTEM_PRICE_PAT = re.compile(r"^システムプライス")
_AREA_PRICE_PAT = re.compile(r"^エリアプライス(" + "|".join(AREA_LABELS_JA) + r")")
_VOLUME_PATS: dict[str, re.Pattern[str]] = {
    "sell_bid_kwh": re.compile(r"^売り?入札量"),
    "buy_bid_kwh": re.compile(r"^買い?入札量"),
    "contract_kwh": re.compile(r"^約定総量"),
}


class SpotCsvFormatError(ValueError):
    """Raised when the CSV no longer looks like a JEPX spot summary."""


def decode(raw: bytes) -> str:
    """Decode JEPX bytes, preferring Shift_JIS but tolerating UTF-8 exports."""
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise SpotCsvFormatError(
        f"could not decode the file as any of {', '.join(ENCODINGS)}"
    )


def normalise_header(name: str) -> str:
    """Collapse a header to a comparable form (NFKC, no spaces, no units)."""
    text = unicodedata.normalize("NFKC", str(name)).strip()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"\(.*?\)", "", text)


def parse_spot_summary(raw: bytes | str, source: str = "") -> SpotFrame:
    """Parse a spot summary CSV into a :class:`SpotFrame`."""
    text = decode(raw) if isinstance(raw, bytes) else raw
    frame = pd.read_csv(StringIO(text), dtype=str, skip_blank_lines=True)
    if frame.empty:
        raise SpotCsvFormatError(f"{source or 'file'} contained a header but no rows")

    headers = {column: normalise_header(column) for column in frame.columns}
    date_col = _find(headers, _DATE_PAT)
    slot_col = _find(headers, _SLOT_PAT)
    system_col = _find(headers, _SYSTEM_PRICE_PAT)
    if not (date_col and slot_col and system_col):
        missing = [
            label
            for label, found in (
                ("delivery date", date_col),
                ("slot code", slot_col),
                ("system price", system_col),
            )
            if not found
        ]
        raise SpotCsvFormatError(
            f"{source or 'file'} is missing required column(s): {', '.join(missing)}. "
            f"Headers seen: {list(frame.columns)}"
        )

    tidy = pd.DataFrame(
        {
            "delivery_date": _parse_dates(frame[date_col], source),
            "slot": _parse_slots(frame[slot_col], source),
        }
    )
    tidy["ts_jst"] = slot_starts(tidy["delivery_date"], tidy["slot"])
    tidy["delivery_date"] = pd.to_datetime(tidy["delivery_date"]).dt.date

    price_cols: dict[str, str] = {system_col: "system"}
    for column, header in headers.items():
        match = _AREA_PRICE_PAT.match(header)
        if match:
            price_cols[column] = AREA_LABELS_JA[match.group(1)]

    prices = tidy.copy()
    for column, area in price_cols.items():
        prices[area] = _to_float(frame[column])
    prices = (
        prices.melt(
            id_vars=["delivery_date", "slot", "ts_jst"],
            value_vars=list(price_cols.values()),
            var_name="area",
            value_name="price_jpy_kwh",
        )
        .dropna(subset=["price_jpy_kwh"])
        .sort_values(["delivery_date", "slot", "area"], ignore_index=True)
    )

    volumes = tidy.copy()
    for name, pattern in _VOLUME_PATS.items():
        column = _find(headers, pattern)
        volumes[name] = _to_float(frame[column]) if column else pd.NA
    volumes = volumes.sort_values(["delivery_date", "slot"], ignore_index=True)

    return SpotFrame(prices=prices, volumes=volumes, source=source)


def slot_count_anomalies(prices: pd.DataFrame) -> pd.DataFrame:
    """Days that do not carry the expected 48 slots for every area.

    Japan has no DST, so a short day means a truncated download or a JEPX
    publication incident.  Reported rather than raised: a partial day is
    still worth storing, it just must not be silently treated as complete.
    """
    counts = (
        prices.groupby(["delivery_date", "area"], as_index=False)["slot"]
        .nunique()
        .rename(columns={"slot": "slots"})
    )
    return counts[counts["slots"] != SLOTS_PER_DAY].reset_index(drop=True)


def _find(headers: dict[str, str], pattern: re.Pattern[str]) -> str | None:
    for column, header in headers.items():
        if pattern.search(header):
            return column
    return None


def _to_float(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"": None, "-": None, "－": None, "nan": None, "None": None})
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_dates(series: pd.Series, source: str) -> pd.Series:
    parsed = pd.to_datetime(series.str.strip(), errors="coerce", format="mixed")
    if parsed.isna().any():
        bad = series[parsed.isna()].head(3).tolist()
        raise SpotCsvFormatError(f"{source or 'file'} has unparseable delivery dates: {bad}")
    return parsed


def _parse_slots(series: pd.Series, source: str) -> pd.Series:
    parsed = pd.to_numeric(series.str.strip(), errors="coerce")
    if parsed.isna().any():
        bad = series[parsed.isna()].head(3).tolist()
        raise SpotCsvFormatError(f"{source or 'file'} has unparseable slot codes: {bad}")
    out_of_range = parsed[(parsed < 1) | (parsed > SLOTS_PER_DAY)]
    if not out_of_range.empty:
        raise SpotCsvFormatError(
            f"{source or 'file'} has slot codes outside 1..{SLOTS_PER_DAY}: "
            f"{out_of_range.head(3).tolist()}"
        )
    return parsed.astype("int16")
