"""Battery physics and the cost of using it.

Degradation is a first-class parameter here rather than an afterthought,
because without it an optimiser answers "cycle hard, every single day" and
the answer is wrong.  With it, the model produces the number an operator
actually needs: the price spread below which the battery should stay
still.

The convention throughout: ``degradation_cost_jpy_per_kwh`` is charged per
kWh *discharged at the grid/AC side*, which is the same basis as the
円/kWh price spread it is compared against.  :meth:`BatterySpec.from_capex`
converts a capex-and-cycle-life quote into that number.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from jepx_arb.timeslot import SLOT_HOURS


@dataclass(frozen=True)
class BatterySpec:
    """A battery system as the dispatch model sees it."""

    name: str
    capacity_kwh: float
    """Nameplate energy capacity at the cells."""

    power_kw: float
    """PCS rating, used for both directions unless overridden below."""

    charge_power_kw: float | None = None
    discharge_power_kw: float | None = None

    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    """One-way efficiencies; their product is the round trip."""

    soc_min: float = 0.05
    soc_max: float = 0.95
    """Usable window as a fraction of ``capacity_kwh``."""

    soc_initial: float = 0.05

    degradation_cost_jpy_per_kwh: float = 0.0
    """Cost per kWh discharged (AC side).  See module docstring."""

    max_cycles_per_day: float | None = 1.0
    """Warranty/O&M limit on equivalent full cycles per day; ``None`` = free."""

    allow_export: bool = True
    """False for the many residential systems that cannot reverse-flow."""

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if self.power_kw <= 0:
            raise ValueError("power_kw must be positive")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("efficiencies must be in (0, 1]")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError("need 0 <= soc_min < soc_max <= 1")
        if not self.soc_min <= self.soc_initial <= self.soc_max:
            raise ValueError("soc_initial must sit inside [soc_min, soc_max]")

    @property
    def charge_kw(self) -> float:
        return self.charge_power_kw or self.power_kw

    @property
    def discharge_kw(self) -> float:
        return self.discharge_power_kw or self.power_kw

    @property
    def usable_kwh(self) -> float:
        """Energy actually cyclable between the SOC limits."""
        return self.capacity_kwh * (self.soc_max - self.soc_min)

    @property
    def round_trip_efficiency(self) -> float:
        return self.charge_efficiency * self.discharge_efficiency

    @property
    def soc_min_kwh(self) -> float:
        return self.capacity_kwh * self.soc_min

    @property
    def soc_max_kwh(self) -> float:
        return self.capacity_kwh * self.soc_max

    @property
    def soc_initial_kwh(self) -> float:
        return self.capacity_kwh * self.soc_initial

    @property
    def max_charge_kwh_per_slot(self) -> float:
        return self.charge_kw * SLOT_HOURS

    @property
    def max_discharge_kwh_per_slot(self) -> float:
        return self.discharge_kw * SLOT_HOURS

    @property
    def ac_kwh_per_full_cycle(self) -> float:
        """Grid-side energy delivered by one full discharge of the usable window."""
        return self.usable_kwh * self.discharge_efficiency

    def breakeven_spread_jpy_kwh(self, buy_price_jpy_kwh: float) -> float:
        """Sell-minus-buy spread below which a cycle cannot pay for itself.

        Delivering 1 kWh to the grid means buying ``1/round_trip`` kWh, so
        the efficiency penalty scales with what the energy cost; the wear
        does not.  This is the single number an operator will quote back at
        you, so it is worth having on the object.
        """
        efficiency_penalty = buy_price_jpy_kwh * (1.0 / self.round_trip_efficiency - 1.0)
        return efficiency_penalty + self.degradation_cost_jpy_per_kwh

    def with_(self, **changes) -> BatterySpec:
        """Copy with overrides -- how the CLI applies per-run tweaks."""
        return replace(self, **changes)

    @staticmethod
    def cost_per_ac_kwh(
        capex_jpy: float,
        cycle_life: int,
        capacity_kwh: float,
        depth_of_discharge: float = 0.9,
        discharge_efficiency: float = 0.95,
        residual_value_jpy: float = 0.0,
    ) -> float:
        """Levelised wear cost per kWh discharged, from a vendor quote.

        ``cycle_life`` full cycles at ``depth_of_discharge`` deliver
        ``cycle_life * capacity * dod * eta`` kWh to the grid over the
        asset's life; the capex spreads across exactly that.
        """
        lifetime_ac_kwh = cycle_life * capacity_kwh * depth_of_discharge * discharge_efficiency
        if lifetime_ac_kwh <= 0:
            raise ValueError("cycle_life, capacity_kwh and depth_of_discharge must be positive")
        return (capex_jpy - residual_value_jpy) / lifetime_ac_kwh

    @classmethod
    def from_capex(
        cls,
        name: str,
        capacity_kwh: float,
        power_kw: float,
        capex_jpy: float,
        cycle_life: int,
        **kwargs,
    ) -> BatterySpec:
        """Build a spec whose wear cost is derived from a capex quote."""
        spec = cls(name=name, capacity_kwh=capacity_kwh, power_kw=power_kw, **kwargs)
        return spec.with_(
            degradation_cost_jpy_per_kwh=cls.cost_per_ac_kwh(
                capex_jpy=capex_jpy,
                cycle_life=cycle_life,
                capacity_kwh=capacity_kwh,
                depth_of_discharge=spec.soc_max - spec.soc_min,
                discharge_efficiency=spec.discharge_efficiency,
            )
        )
