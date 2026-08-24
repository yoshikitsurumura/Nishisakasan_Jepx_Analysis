"""Data sources.

Every source implements :class:`~jepx_arb.sources.base.PriceSource` so the
rest of the project never depends on *where* prices came from.  Today that
is the public JEPX CSV; tomorrow it can be a paid API or a file someone
downloaded by hand during an outage.
"""

from jepx_arb.sources.base import AREAS, PriceSource, SpotFrame
from jepx_arb.sources.jepx_csv import JepxCsvSource
from jepx_arb.sources.manual import ManualCsvSource

__all__ = ["AREAS", "PriceSource", "SpotFrame", "JepxCsvSource", "ManualCsvSource"]
