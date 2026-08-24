"""Download ``spot_summary_YYYY.csv`` from the public JEPX site.

The market-data pages render their tables with JavaScript, so scraping the
HTML is both fragile and unnecessary: the same page's download button posts
to ``_download.php`` and returns the CSV directly.  That endpoint is what
this module talks to.

Two things it insists on, because the endpoint rejects requests without
them: a ``Referer`` pointing at the spot page, and a POST body naming the
directory and file.  Everything else here is about failing loudly and
retrying politely -- the whole file is one request per day, so there is no
reason to be anything but gentle with it.

Attribution: any output derived from this data must credit JEPX
(see :data:`ATTRIBUTION`).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import requests

from jepx_arb.sources.base import SpotFrame
from jepx_arb.sources.parser import parse_spot_summary

log = logging.getLogger(__name__)

DOWNLOAD_URL = "https://www.jepx.jp/_download.php"
SPOT_PAGE = "https://www.jepx.jp/electricpower/market-data/spot/"
SPOT_DIR = "spot_summary"

ATTRIBUTION = "出典：一般社団法人日本卸電力取引所 (JEPX)"

USER_AGENT = (
    "jepx-arb-lab/0.1 (+https://github.com/yoshikitsurumura/Nishisakasan_Jepx_Analysis) "
    "python-requests"
)


class JepxDownloadError(RuntimeError):
    """The download did not return something that looks like a CSV."""


@dataclass
class JepxCsvSource:
    """Fetch and parse the yearly spot summary file.

    ``session`` is injectable so tests -- and any future move to a paid
    feed -- do not have to touch the network.
    """

    name: str = "jepx_csv"
    timeout: float = 60.0
    retries: int = 4
    backoff: float = 2.0
    session: requests.Session | None = None
    _owned_session: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
            self._owned_session = True

    @staticmethod
    def filename(year: int) -> str:
        return f"spot_summary_{year}.csv"

    def fetch_year(self, year: int) -> SpotFrame:
        """Download one year's file and parse it."""
        raw = self.download_year(year)
        return parse_spot_summary(raw, source=self.filename(year))

    def download_year(self, year: int) -> bytes:
        """Return the raw CSV bytes, retrying transient network failures."""
        filename = self.filename(year)
        headers = {
            "Referer": SPOT_PAGE,
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/csv,application/octet-stream,*/*",
        }
        payload = {"dir": SPOT_DIR, "file": filename}

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            params = {"timestamp": str(int(time.time() * 1000))}
            try:
                response = self.session.post(  # type: ignore[union-attr]
                    DOWNLOAD_URL,
                    params=params,
                    data=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return _validate(response.content, filename)
            except (requests.RequestException, JepxDownloadError) as error:
                last_error = error
                if attempt == self.retries:
                    break
                delay = self.backoff ** attempt
                log.warning(
                    "JEPX download of %s failed (attempt %d/%d): %s -- retrying in %.0fs",
                    filename,
                    attempt,
                    self.retries,
                    error,
                    delay,
                )
                time.sleep(delay)

        raise JepxDownloadError(
            f"could not download {filename} after {self.retries} attempts: {last_error}"
        ) from last_error

    def close(self) -> None:
        if self._owned_session and self.session is not None:
            self.session.close()


def _validate(content: bytes, filename: str) -> bytes:
    """Reject error pages served with a 200, which is how this endpoint fails."""
    if len(content) < 200:
        raise JepxDownloadError(f"{filename} came back as only {len(content)} bytes")
    head = content[:512].lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        raise JepxDownloadError(
            f"{filename} came back as HTML, not CSV -- the download endpoint or its "
            "required headers have probably changed"
        )
    return content
