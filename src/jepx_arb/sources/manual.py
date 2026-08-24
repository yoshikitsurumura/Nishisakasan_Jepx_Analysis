"""Read a spot summary CSV from disk.

This is the escape hatch the project is designed around.  When the public
endpoint changes shape or goes away, someone can still download the file
by hand from the JEPX site and feed it in through exactly the same
interface, and nothing downstream notices the difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jepx_arb.sources.base import SpotFrame
from jepx_arb.sources.parser import parse_spot_summary


@dataclass
class ManualCsvSource:
    """Load ``spot_summary_YYYY.csv`` files from a local directory."""

    directory: Path
    name: str = "manual_csv"

    def path_for(self, year: int) -> Path:
        return Path(self.directory) / f"spot_summary_{year}.csv"

    def fetch_year(self, year: int) -> SpotFrame:
        path = self.path_for(year)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- download the yearly CSV from "
                "https://www.jepx.jp/electricpower/market-data/spot/ and put it there"
            )
        return parse_spot_summary(path.read_bytes(), source=str(path))

    def fetch_file(self, path: Path) -> SpotFrame:
        """Parse one specific file, whatever it is named."""
        return parse_spot_summary(Path(path).read_bytes(), source=str(path))
