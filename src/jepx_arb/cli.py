"""Command line entry point.

Four verbs:

``fetch``    pull a year of spot prices into the Parquet store
``stats``    describe what is in the store -- price spreads, coverage
``backtest`` perfect-foresight profit for one battery/tariff/area
``compare``  the same run across presets and areas, ranked

The one that matters first is ``compare``: it puts every customer segment
in a single table before any hardware has been chosen.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from jepx_arb import presets
from jepx_arb.backtest.engine import BacktestConfig, run_backtest
from jepx_arb.backtest.report import (
    comparison_table,
    format_comparison,
    format_report,
    summarise,
)
from jepx_arb.battery import BatterySpec
from jepx_arb.sources.base import DELIVERY_AREAS
from jepx_arb.sources.jepx_csv import JepxCsvSource
from jepx_arb.sources.manual import ManualCsvSource
from jepx_arb.sources.parser import slot_count_anomalies
from jepx_arb.sources.store import SpotStore

log = logging.getLogger("jepx_arb")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jepx-arb", description="JEPX spot ingestion and battery arbitrage backtesting"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/spot"), help="Parquet store location"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="download spot prices into the store")
    fetch.add_argument(
        "--year",
        type=int,
        nargs="*",
        help="JEPX file year(s); defaults to the current and previous calendar year",
    )
    fetch.add_argument(
        "--from-file",
        type=Path,
        help="parse a locally downloaded CSV instead of hitting the network",
    )
    fetch.add_argument(
        "--raw-dir", type=Path, help="also keep the untouched download here"
    )
    fetch.set_defaults(handler=_cmd_fetch)

    stats = subparsers.add_parser("stats", help="describe what the store holds")
    stats.add_argument("--area", default="tokyo", choices=list(DELIVERY_AREAS) + ["system"])
    stats.add_argument("--from", dest="start", help="YYYY-MM-DD")
    stats.add_argument("--to", dest="end", help="YYYY-MM-DD")
    stats.set_defaults(handler=_cmd_stats)

    backtest = subparsers.add_parser("backtest", help="perfect-foresight profit for one setup")
    _add_run_arguments(backtest)
    backtest.add_argument("--preset", default="grid-2mwh", choices=presets.keys())
    backtest.add_argument("--area", default="tokyo", choices=list(DELIVERY_AREAS) + ["system"])
    backtest.add_argument("--schedule-out", type=Path, help="write the per-slot dispatch to CSV")
    backtest.add_argument("--daily-out", type=Path, help="write per-day results to CSV")
    backtest.add_argument("--json", action="store_true", help="emit the summary as JSON")
    backtest.set_defaults(handler=_cmd_backtest)

    compare = subparsers.add_parser("compare", help="rank presets and areas side by side")
    _add_run_arguments(compare)
    compare.add_argument("--presets", nargs="*", default=presets.keys(), choices=presets.keys())
    compare.add_argument(
        "--areas", nargs="*", default=["tokyo"], choices=list(DELIVERY_AREAS) + ["system"]
    )
    compare.add_argument("--csv-out", type=Path, help="write the comparison table to CSV")
    compare.set_defaults(handler=_cmd_compare)

    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--from", dest="start", help="YYYY-MM-DD")
    parser.add_argument("--to", dest="end", help="YYYY-MM-DD")
    parser.add_argument(
        "--horizon",
        default="day",
        choices=["day", "continuous"],
        help="solve each day separately (realistic) or the whole window at once (outer bound)",
    )
    parser.add_argument(
        "--terminal",
        default="initial",
        choices=["initial", "atleast", "free"],
        help="end-of-horizon state of charge",
    )
    parser.add_argument("--capacity", type=float, help="override capacity in kWh")
    parser.add_argument("--power", type=float, help="override PCS rating in kW")
    parser.add_argument(
        "--degradation", type=float, help="override wear cost in JPY per kWh discharged"
    )
    parser.add_argument(
        "--capex", type=float, help="re-derive wear cost from this capex (needs --cycle-life)"
    )
    parser.add_argument("--cycle-life", type=int, help="full cycles the capex is spread over")
    parser.add_argument(
        "--max-cycles", type=float, help="cap on equivalent full cycles per day"
    )
    parser.add_argument("--load-kwh", type=float, help="override daily site load in kWh")


def _cmd_fetch(args: argparse.Namespace) -> int:
    store = SpotStore(args.data_dir)

    if args.from_file:
        source = ManualCsvSource(directory=args.from_file.parent)
        frame = source.fetch_file(args.from_file)
        raw = args.from_file.read_bytes()
        first, _ = frame.date_range
        year = first.year if first else dt.date.today().year
        print(store.write(frame, year, raw).describe())
        _report_anomalies(frame)
        return 0

    today = dt.date.today()
    years = args.year or sorted({today.year - 1, today.year})
    source = JepxCsvSource()
    failures = 0
    try:
        for year in years:
            try:
                raw = source.download_year(year)
            except Exception as error:  # noqa: BLE001 - one bad year must not stop the rest
                log.error("fetch %d failed: %s", year, error)
                failures += 1
                continue
            if args.raw_dir:
                args.raw_dir.mkdir(parents=True, exist_ok=True)
                (args.raw_dir / source.filename(year)).write_bytes(raw)
            frame = _parse(raw, source.filename(year))
            print(store.write(frame, year, raw).describe())
            _report_anomalies(frame)
    finally:
        source.close()
    return 1 if failures == len(years) else 0


def _parse(raw: bytes, name: str):
    from jepx_arb.sources.parser import parse_spot_summary

    return parse_spot_summary(raw, source=name)


def _cmd_stats(args: argparse.Namespace) -> int:
    store = SpotStore(args.data_dir)
    prices = store.load(start=args.start, end=args.end, areas=[args.area])
    if prices.empty:
        print("Store is empty for that area and range.")
        return 1

    daily = prices.groupby("delivery_date")["price_jpy_kwh"].agg(["min", "max", "mean", "count"])
    daily["spread"] = daily["max"] - daily["min"]
    complete = daily[daily["count"] == 48]

    print(f"area {args.area}: {len(daily)} days, {len(prices):,} slots")
    print(f"  range           {daily.index.min()} .. {daily.index.max()}")
    print(f"  complete days   {len(complete)} of {len(daily)}")
    print(f"  price           mean {prices['price_jpy_kwh'].mean():.2f}  "
          f"min {prices['price_jpy_kwh'].min():.2f}  max {prices['price_jpy_kwh'].max():.2f}")
    print(f"  daily spread    mean {daily['spread'].mean():.2f}  "
          f"median {daily['spread'].median():.2f}  max {daily['spread'].max():.2f}")
    print("  spread percentiles (JPY/kWh)")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"    p{int(q * 100):<3}          {daily['spread'].quantile(q):.2f}")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    store = SpotStore(args.data_dir)
    prices = store.load(start=args.start, end=args.end, areas=[args.area])
    preset = presets.get(args.preset)
    config = _config_from(args, preset, args.area, keep_schedule=bool(args.schedule_out))
    result = run_backtest(prices, config)

    if args.json:
        print(json.dumps(summarise(result), indent=2, default=str))
    else:
        print(format_report(result))

    if args.daily_out and len(result.daily):
        _write_csv(result.daily, args.daily_out)
    if args.schedule_out and result.schedule is not None:
        _write_csv(result.schedule, args.schedule_out)
    return 0 if result.days else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    store = SpotStore(args.data_dir)
    results = []
    for area in args.areas:
        prices = store.load(start=args.start, end=args.end, areas=[area])
        for key in args.presets:
            preset = presets.get(key)
            config = _config_from(args, preset, area, keep_schedule=False)
            results.append(run_backtest(prices, config))

    print(format_comparison(results))
    if args.csv_out:
        _write_csv(comparison_table(results), args.csv_out)
    return 0


def _config_from(
    args: argparse.Namespace, preset: presets.Preset, area: str, keep_schedule: bool
) -> BacktestConfig:
    spec = preset.spec
    overrides: dict = {}
    if args.capacity is not None:
        overrides["capacity_kwh"] = args.capacity
    if args.power is not None:
        overrides["power_kw"] = args.power
    if args.max_cycles is not None:
        overrides["max_cycles_per_day"] = args.max_cycles
    if overrides:
        spec = spec.with_(**overrides)

    if args.capex is not None or args.cycle_life is not None:
        if args.capex is None or args.cycle_life is None:
            raise SystemExit("--capex and --cycle-life must be given together")
        spec = spec.with_(
            degradation_cost_jpy_per_kwh=BatterySpec.cost_per_ac_kwh(
                capex_jpy=args.capex,
                cycle_life=args.cycle_life,
                capacity_kwh=spec.capacity_kwh,
                depth_of_discharge=spec.soc_max - spec.soc_min,
                discharge_efficiency=spec.discharge_efficiency,
            )
        )
    if args.degradation is not None:
        spec = spec.with_(degradation_cost_jpy_per_kwh=args.degradation)

    return BacktestConfig(
        spec=spec,
        tariff=preset.tariff,
        area=area,
        horizon=args.horizon,
        terminal=args.terminal,
        load_profile=preset.load_profile,
        load_daily_kwh=(
            args.load_kwh if args.load_kwh is not None else preset.load_daily_kwh
        ),
        keep_schedule=keep_schedule,
    )


def _report_anomalies(frame) -> None:
    anomalies = slot_count_anomalies(frame.prices)
    if len(anomalies):
        log.warning(
            "%d date/area pairs do not have 48 slots (first: %s)",
            len(anomalies),
            anomalies.iloc[0].to_dict(),
        )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path} ({len(frame):,} rows)")


if __name__ == "__main__":
    sys.exit(main())
