"""End-to-end CLI runs, still without touching the network."""

from __future__ import annotations

import json

import pytest

from jepx_arb.cli import main


@pytest.fixture
def loaded_store(tmp_path, sample_csv_bytes):
    """A store populated through the CLI's own manual-import path."""
    csv = tmp_path / "spot_summary_2024.csv"
    csv.write_bytes(sample_csv_bytes)
    store = tmp_path / "store"
    assert main(["--data-dir", str(store), "fetch", "--from-file", str(csv)]) == 0
    return store


def test_fetch_from_file_populates_the_store(loaded_store):
    assert (loaded_store / "spot_prices_2024.parquet").exists()
    assert (loaded_store / "manifest.json").exists()


def test_fetch_reports_the_incomplete_day(tmp_path, sample_csv_bytes, caplog):
    csv = tmp_path / "spot_summary_2024.csv"
    csv.write_bytes(sample_csv_bytes)
    with caplog.at_level("WARNING"):
        main(["--data-dir", str(tmp_path / "s"), "fetch", "--from-file", str(csv)])
    assert "do not have 48 slots" in caplog.text


def test_stats_describes_the_spread(loaded_store, capsys):
    assert main(["--data-dir", str(loaded_store), "stats", "--area", "tokyo"]) == 0
    output = capsys.readouterr().out
    assert "daily spread" in output
    assert "complete days   3 of 3" in output


def test_backtest_prints_a_report(loaded_store, capsys):
    assert main(["--data-dir", str(loaded_store), "backtest", "--preset", "grid-2mwh"]) == 0
    output = capsys.readouterr().out
    assert "per kWh of capacity" in output
    assert "Perfect foresight" in output


def test_backtest_json_is_machine_readable(loaded_store, capsys):
    main(["--data-dir", str(loaded_store), "backtest", "--preset", "grid-2mwh", "--json"])
    stats = json.loads(capsys.readouterr().out)
    assert stats["days"] == 3
    assert stats["area"] == "tokyo"


def test_backtest_writes_the_schedule_when_asked(loaded_store, tmp_path, capsys):
    out = tmp_path / "out" / "schedule.csv"
    main([
        "--data-dir", str(loaded_store), "backtest",
        "--preset", "grid-2mwh", "--schedule-out", str(out),
    ])
    assert out.exists()
    assert "soc_kwh" in out.read_text().splitlines()[0]


def test_overrides_change_the_answer(loaded_store, capsys):
    main(["--data-dir", str(loaded_store), "backtest", "--preset", "grid-2mwh",
          "--degradation", "0", "--json"])
    free = json.loads(capsys.readouterr().out)
    main(["--data-dir", str(loaded_store), "backtest", "--preset", "grid-2mwh",
          "--degradation", "500", "--json"])
    expensive = json.loads(capsys.readouterr().out)
    assert free["net_profit_jpy"] > expensive["net_profit_jpy"]
    assert expensive["net_profit_jpy"] == pytest.approx(0.0, abs=1e-6)


def test_capex_needs_a_cycle_life(loaded_store):
    with pytest.raises(SystemExit, match="must be given together"):
        main(["--data-dir", str(loaded_store), "backtest", "--capex", "1000000"])


def test_compare_ranks_every_preset(loaded_store, capsys):
    assert main([
        "--data-dir", str(loaded_store), "compare",
        "--areas", "tokyo", "kansai",
    ]) == 0
    output = capsys.readouterr().out
    assert "JPY/kWh/yr" in output
    assert output.count("grid-2mwh") == 2
    assert "home-8kwh" in output


def test_compare_writes_a_csv(loaded_store, tmp_path):
    out = tmp_path / "compare.csv"
    main(["--data-dir", str(loaded_store), "compare", "--areas", "tokyo", "--csv-out", str(out)])
    assert "jpy_per_kwh_capacity_year" in out.read_text().splitlines()[0]


def test_stats_on_an_empty_range_exits_nonzero(loaded_store, capsys):
    assert main([
        "--data-dir", str(loaded_store), "stats",
        "--from", "2030-01-01", "--to", "2030-01-02",
    ]) == 1
