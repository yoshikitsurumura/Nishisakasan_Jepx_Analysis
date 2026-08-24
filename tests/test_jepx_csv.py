"""The HTTP layer, exercised without touching the network."""

from __future__ import annotations

import pytest
import requests

from jepx_arb.sources.jepx_csv import SPOT_PAGE, JepxCsvSource, JepxDownloadError


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """Replays a queued list of responses (or exceptions) and records calls."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, params=None, data=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "params": params, "data": data, "headers": headers, "timeout": timeout}
        )
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self):
        pass


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr("jepx_arb.sources.jepx_csv.time.sleep", lambda _: None)


def test_sends_the_referer_the_endpoint_requires(sample_csv_bytes):
    session = FakeSession(FakeResponse(sample_csv_bytes))
    JepxCsvSource(session=session).download_year(2025)
    call = session.calls[0]
    assert call["headers"]["Referer"] == SPOT_PAGE
    assert call["data"] == {"dir": "spot_summary", "file": "spot_summary_2025.csv"}


def test_cache_busting_timestamp_is_sent(sample_csv_bytes):
    session = FakeSession(FakeResponse(sample_csv_bytes))
    JepxCsvSource(session=session).download_year(2025)
    assert session.calls[0]["params"]["timestamp"].isdigit()


def test_fetch_year_parses_what_it_downloads(sample_csv_bytes):
    source = JepxCsvSource(session=FakeSession(FakeResponse(sample_csv_bytes)))
    frame = source.fetch_year(2024)
    assert frame.source == "spot_summary_2024.csv"
    assert len(frame.prices) == 3 * 48 * 10 - 1


def test_html_error_page_is_not_mistaken_for_data():
    """This endpoint answers failures with a 200 and an HTML body, which is
    exactly how a silent data-corruption bug gets in."""
    html = b"<!DOCTYPE html><html><body>" + b"x" * 300 + b"</body></html>"
    source = JepxCsvSource(session=FakeSession(*[FakeResponse(html)] * 4), retries=4)
    with pytest.raises(JepxDownloadError, match="came back as HTML"):
        source.download_year(2025)


def test_suspiciously_short_body_is_rejected():
    source = JepxCsvSource(session=FakeSession(*[FakeResponse(b"oops")] * 2), retries=2)
    with pytest.raises(JepxDownloadError, match="only 4 bytes"):
        source.download_year(2025)


def test_transient_failure_is_retried_then_succeeds(sample_csv_bytes):
    session = FakeSession(
        requests.ConnectionError("reset"),
        FakeResponse(b"", 503),
        FakeResponse(sample_csv_bytes),
    )
    content = JepxCsvSource(session=session, retries=4).download_year(2025)
    assert content == sample_csv_bytes
    assert len(session.calls) == 3


def test_gives_up_after_the_configured_attempts():
    session = FakeSession(*[requests.ConnectionError("down")] * 3)
    with pytest.raises(JepxDownloadError, match="after 3 attempts"):
        JepxCsvSource(session=session, retries=3).download_year(2025)
    assert len(session.calls) == 3
