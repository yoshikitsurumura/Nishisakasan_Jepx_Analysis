"""Parquet store for normalised spot data.

Deliberately not a database.  Ten areas x 48 slots x 365 days is roughly
175k price rows a year, a couple of megabytes of Parquet, which DuckDB or
pandas will query directly.  Keeping it as files in the repo means the
data has a git history: when JEPX republishes a corrected day, the diff
shows exactly what moved.  Move to Postgres when the size argues for it,
not before.

Writes are upserts keyed on ``(delivery_date, slot, area)`` -- the yearly
file is re-downloaded whole every run, so the store has to fold a fresh
copy of the whole year into what it already has, letting corrections win.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from jepx_arb.sources.base import PRICE_COLUMNS, VOLUME_COLUMNS, SpotFrame

PRICE_KEY = ["delivery_date", "slot", "area"]
VOLUME_KEY = ["delivery_date", "slot"]


@dataclass(frozen=True)
class WriteResult:
    """What one ingest actually changed, so a cron run can report itself."""

    year: int
    rows_in: int
    rows_total: int
    rows_added: int
    rows_changed: int

    @property
    def changed(self) -> bool:
        return bool(self.rows_added or self.rows_changed)

    def describe(self) -> str:
        if not self.changed:
            return f"{self.year}: no change ({self.rows_total} rows on file)"
        return (
            f"{self.year}: +{self.rows_added} new, {self.rows_changed} revised "
            f"({self.rows_total} rows on file)"
        )


@dataclass
class SpotStore:
    """Read and write the yearly Parquet files under ``root``."""

    root: Path = Path("data/spot")

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def price_path(self, year: int) -> Path:
        return self.root / f"spot_prices_{year}.parquet"

    def volume_path(self, year: int) -> Path:
        return self.root / f"spot_volumes_{year}.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def years(self) -> list[int]:
        return sorted(
            int(path.stem.rsplit("_", 1)[1]) for path in self.root.glob("spot_prices_*.parquet")
        )

    def write(self, frame: SpotFrame, year: int, raw: bytes | None = None) -> WriteResult:
        """Fold ``frame`` into the store, letting newer rows win."""
        self.root.mkdir(parents=True, exist_ok=True)

        existing = self._read_parquet(self.price_path(year), PRICE_COLUMNS)
        merged, added, changed = _upsert(existing, frame.prices, PRICE_KEY, "price_jpy_kwh")
        merged.to_parquet(self.price_path(year), index=False)

        old_volumes = self._read_parquet(self.volume_path(year), VOLUME_COLUMNS)
        volumes, _, _ = _upsert(old_volumes, frame.volumes, VOLUME_KEY, "contract_kwh")
        volumes.to_parquet(self.volume_path(year), index=False)

        result = WriteResult(
            year=year,
            rows_in=len(frame.prices),
            rows_total=len(merged),
            rows_added=added,
            rows_changed=changed,
        )
        self._update_manifest(result, frame, raw)
        return result

    def load(
        self,
        start: dt.date | str | None = None,
        end: dt.date | str | None = None,
        areas: list[str] | str | None = None,
    ) -> pd.DataFrame:
        """Return long-format prices for a date range, sorted."""
        years = self.years()
        if not years:
            raise FileNotFoundError(
                f"no Parquet files under {self.root} -- run `jepx-arb fetch` first"
            )
        start_date = _as_date(start)
        end_date = _as_date(end)
        wanted = [
            year
            for year in years
            # JEPX file years are fiscal-ish at the edges, so keep a year of slack.
            if (start_date is None or year >= start_date.year - 1)
            and (end_date is None or year <= end_date.year + 1)
        ]
        parts = [pd.read_parquet(self.price_path(year)) for year in wanted]
        if not parts:
            return pd.DataFrame(columns=list(PRICE_COLUMNS))
        prices = pd.concat(parts, ignore_index=True)

        if start_date is not None:
            prices = prices[prices["delivery_date"] >= start_date]
        if end_date is not None:
            prices = prices[prices["delivery_date"] <= end_date]
        if areas is not None:
            wanted_areas = [areas] if isinstance(areas, str) else list(areas)
            prices = prices[prices["area"].isin(wanted_areas)]
        return prices.sort_values(PRICE_KEY, ignore_index=True)

    def load_wide(self, **kwargs) -> pd.DataFrame:
        """Prices pivoted to one column per area."""
        prices = self.load(**kwargs)
        return (
            prices.pivot_table(
                index=["delivery_date", "slot"], columns="area", values="price_jpy_kwh"
            )
            .rename_axis(columns=None)
            .sort_index()
        )

    def _read_parquet(self, path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=list(columns))
        return pd.read_parquet(path)

    def _update_manifest(self, result: WriteResult, frame: SpotFrame, raw: bytes | None) -> None:
        manifest = {}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        first, last = frame.date_range
        manifest[str(result.year)] = {
            "fetched_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "source": frame.source,
            "rows_in_file": result.rows_in,
            "rows_stored": result.rows_total,
            "rows_added": result.rows_added,
            "rows_revised": result.rows_changed,
            "date_min": str(first) if first else None,
            "date_max": str(last) if last else None,
            "raw_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _upsert(
    existing: pd.DataFrame, incoming: pd.DataFrame, key: list[str], compare: str
) -> tuple[pd.DataFrame, int, int]:
    """Merge ``incoming`` over ``existing``, counting new and revised rows."""
    if existing.empty:
        return incoming.copy(), len(incoming), 0

    old = existing.set_index(key)
    new = incoming.set_index(key)

    added = int(len(new.index.difference(old.index)))
    shared = old.index.intersection(new.index)
    changed = 0
    if len(shared) and compare in old.columns and compare in new.columns:
        before = old.loc[shared, compare]
        after = new.loc[shared, compare]
        # NaN != NaN, so compare on the pair of "is null" masks too.
        differs = (before != after) & ~(before.isna() & after.isna())
        changed = int(differs.sum())

    merged = pd.concat([old[~old.index.isin(new.index)], new]).reset_index()
    return merged.sort_values(key, ignore_index=True), added, changed


def _as_date(value: dt.date | str | None) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))
