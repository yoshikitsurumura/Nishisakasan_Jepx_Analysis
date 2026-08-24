"""The four segments worth comparing before committing to one.

The point of having these side by side is that "which customer segment"
is usually decided by gut feel and a vendor's spreadsheet.  Running the
same price history through all four turns it into a number: 円 per kWh of
installed capacity per year, which a 8 kWh home unit and a 2 MWh
grid-scale system can be compared on directly.

**Every number below is a placeholder.** Capex, cycle life, wheeling
charges and retail margins are all deal-specific.  They are here so the
tool runs out of the box and the shape of the answer is visible; replace
them with real quotes before the answer is used for anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from jepx_arb.battery import BatterySpec
from jepx_arb.tariff import Tariff


@dataclass(frozen=True)
class Preset:
    """A battery, the tariff it faces, and the load that constrains it."""

    key: str
    label: str
    spec: BatterySpec
    tariff: Tariff
    load_profile: str | None = None
    load_daily_kwh: float = 0.0
    notes: str = ""


def _home_battery(name: str, capex_jpy: float = 1_200_000, cycle_life: int = 12_000):
    return BatterySpec.from_capex(
        name=name,
        capacity_kwh=8.0,
        power_kw=3.0,
        capex_jpy=capex_jpy,
        cycle_life=cycle_life,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        soc_min=0.05,
        soc_max=0.95,
        soc_initial=0.05,
        max_cycles_per_day=1.0,
        allow_export=False,
    )


PRESETS: dict[str, Preset] = {
    "home-8kwh": Preset(
        key="home-8kwh",
        label="家庭用 8kWh / 市場連動プラン / 逆潮流不可",
        spec=_home_battery("home-8kwh"),
        tariff=Tariff.market_linked_retail(adder_jpy_kwh=12.0, can_export=False),
        load_profile="household",
        load_daily_kwh=12.0,
        notes="Capex 1.2M JPY over 12,000 cycles. Discharge is capped by household load.",
    ),
    "home-8kwh-tou": Preset(
        key="home-8kwh-tou",
        label="家庭用 8kWh / 時間帯別プラン / 逆潮流不可",
        spec=_home_battery("home-8kwh-tou"),
        tariff=Tariff.time_of_use(),
        load_profile="household",
        load_daily_kwh=12.0,
        notes="Baseline: a fixed night-discount tariff, which ignores JEPX entirely.",
    ),
    "hv-500kwh": Preset(
        key="hv-500kwh",
        label="高圧自家消費 500kWh / 市場連動 / 逆潮流不可",
        spec=BatterySpec.from_capex(
            name="hv-500kwh",
            capacity_kwh=500.0,
            power_kw=250.0,
            capex_jpy=40_000_000,
            cycle_life=10_000,
            charge_efficiency=0.96,
            discharge_efficiency=0.96,
            soc_min=0.05,
            soc_max=0.95,
            soc_initial=0.05,
            max_cycles_per_day=1.0,
            allow_export=False,
        ),
        tariff=Tariff.market_linked_retail(adder_jpy_kwh=6.0, can_export=False),
        load_profile="commercial",
        load_daily_kwh=3_000.0,
        notes="High-voltage site: a thinner adder than residential, weekday-shaped load.",
    ),
    "grid-2mwh": Preset(
        key="grid-2mwh",
        label="系統用 2MWh / 卸電力市場に直接参加",
        spec=BatterySpec.from_capex(
            name="grid-2mwh",
            capacity_kwh=2_000.0,
            power_kw=1_000.0,
            capex_jpy=120_000_000,
            cycle_life=10_000,
            charge_efficiency=0.96,
            discharge_efficiency=0.96,
            soc_min=0.05,
            soc_max=0.95,
            soc_initial=0.05,
            max_cycles_per_day=1.0,
            allow_export=True,
        ),
        tariff=Tariff.wholesale(),
        notes=(
            "Spot arbitrage only. Real projects earn most of their revenue in the "
            "capacity and balancing markets, which this model does not include."
        ),
    ),
}


def get(key: str) -> Preset:
    if key not in PRESETS:
        raise KeyError(f"unknown preset {key!r}; available: {', '.join(PRESETS)}")
    return PRESETS[key]


def keys() -> list[str]:
    return list(PRESETS)
