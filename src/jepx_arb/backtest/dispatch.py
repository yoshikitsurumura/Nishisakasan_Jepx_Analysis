"""Optimal battery dispatch against a known price path.

This answers one question only: with perfect hindsight, what was the most
money this battery could have made?  That number is the ceiling every
forecast-driven strategy is measured against later -- without it, an
improvement in forecast accuracy has no units anyone can act on.

The problem is a linear program.  SOC carries as its own variable rather
than being expanded into cumulative sums, which keeps the constraint
matrix at a handful of non-zeros per slot; that is what lets the same code
solve one 48-slot day and a whole continuous year.

Sign convention: ``charge_kwh`` and ``discharge_kwh`` are both positive and
measured at the grid/AC side, so they line up directly with the 円/kWh
prices they multiply.  ``soc_kwh`` is at the cells, end of slot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from jepx_arb.battery import BatterySpec

log = logging.getLogger(__name__)

TerminalMode = str  # "initial" | "free" | "atleast"


class InfeasibleDispatch(RuntimeError):
    """The LP had no solution -- usually contradictory SOC or load limits."""


@dataclass(frozen=True)
class DispatchResult:
    """One solved dispatch problem."""

    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    soc_kwh: np.ndarray
    buy_price: np.ndarray
    sell_price: np.ndarray
    degradation_cost_jpy_per_kwh: float

    @property
    def energy_cost_jpy(self) -> float:
        return float(self.buy_price @ self.charge_kwh)

    @property
    def revenue_jpy(self) -> float:
        return float(self.sell_price @ self.discharge_kwh)

    @property
    def degradation_jpy(self) -> float:
        return float(self.degradation_cost_jpy_per_kwh * self.discharge_kwh.sum())

    @property
    def net_profit_jpy(self) -> float:
        return self.revenue_jpy - self.energy_cost_jpy - self.degradation_jpy

    @property
    def charged_kwh(self) -> float:
        return float(self.charge_kwh.sum())

    @property
    def discharged_kwh(self) -> float:
        return float(self.discharge_kwh.sum())

    def equivalent_full_cycles(self, spec: BatterySpec) -> float:
        """Discharged energy expressed in whole usable-window cycles."""
        return self.discharged_kwh / spec.ac_kwh_per_full_cycle


def optimise(
    spec: BatterySpec,
    buy_price: np.ndarray,
    sell_price: np.ndarray,
    *,
    soc_start_kwh: float | None = None,
    max_discharge_kwh: np.ndarray | None = None,
    max_charge_kwh: np.ndarray | None = None,
    terminal: TerminalMode = "initial",
    cycle_cap: float | None = None,
) -> DispatchResult:
    """Maximise profit over the given price path.

    ``max_discharge_kwh`` is how the no-export case enters the model: pass
    the site load per slot and discharge cannot exceed it.  ``cycle_cap``
    overrides the spec's per-day cycle limit (the engine scales it by the
    number of days when solving a continuous horizon).
    """
    buy = np.asarray(buy_price, dtype=float)
    sell = np.asarray(sell_price, dtype=float)
    if buy.shape != sell.shape or buy.ndim != 1:
        raise ValueError("buy_price and sell_price must be 1-D and the same length")
    horizon = buy.size
    if horizon == 0:
        raise ValueError("cannot optimise an empty horizon")

    soc_start = spec.soc_initial_kwh if soc_start_kwh is None else float(soc_start_kwh)
    if not spec.soc_min_kwh - 1e-9 <= soc_start <= spec.soc_max_kwh + 1e-9:
        raise ValueError(
            f"soc_start_kwh {soc_start:.3f} is outside the usable window "
            f"[{spec.soc_min_kwh:.3f}, {spec.soc_max_kwh:.3f}]"
        )

    _warn_on_free_money(spec, buy, sell)

    charge_cap = _cap(max_charge_kwh, spec.max_charge_kwh_per_slot, horizon)
    discharge_cap = _cap(max_discharge_kwh, spec.max_discharge_kwh_per_slot, horizon)
    discharge_cap = np.minimum(discharge_cap, spec.max_discharge_kwh_per_slot)

    # x = [charge (T) | discharge (T) | soc (T)]
    objective = np.concatenate(
        [buy, -(sell - spec.degradation_cost_jpy_per_kwh), np.zeros(horizon)]
    )

    a_eq, b_eq = _soc_dynamics(spec, horizon, soc_start)

    bounds: list[tuple[float, float]] = []
    bounds.extend((0.0, float(cap)) for cap in charge_cap)
    bounds.extend((0.0, float(cap)) for cap in discharge_cap)
    soc_bounds = [(spec.soc_min_kwh, spec.soc_max_kwh) for _ in range(horizon)]
    soc_bounds[-1] = _terminal_bounds(spec, soc_start, terminal)
    bounds.extend(soc_bounds)

    a_ub, b_ub = _cycle_limit(spec, horizon, cycle_cap)

    solution = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        raise InfeasibleDispatch(
            f"dispatch LP failed ({solution.message.strip()}) over {horizon} slots; "
            "check the SOC window, the load cap and the terminal condition"
        )

    x = np.asarray(solution.x, dtype=float)
    charge = np.clip(x[:horizon], 0.0, None)
    discharge = np.clip(x[horizon : 2 * horizon], 0.0, None)
    soc = x[2 * horizon :]

    return DispatchResult(
        charge_kwh=charge,
        discharge_kwh=discharge,
        soc_kwh=soc,
        buy_price=buy,
        sell_price=sell,
        degradation_cost_jpy_per_kwh=spec.degradation_cost_jpy_per_kwh,
    )


def _soc_dynamics(
    spec: BatterySpec, horizon: int, soc_start: float
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """s_t - s_{t-1} - eta_c * c_t + d_t / eta_d = 0, with s_-1 = soc_start."""
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    for t in range(horizon):
        rows += [t, t, t]
        cols += [t, horizon + t, 2 * horizon + t]
        values += [-spec.charge_efficiency, 1.0 / spec.discharge_efficiency, 1.0]
        if t >= 1:
            rows.append(t)
            cols.append(2 * horizon + t - 1)
            values.append(-1.0)

    a_eq = sparse.csr_matrix(
        (values, (rows, cols)), shape=(horizon, 3 * horizon), dtype=float
    )
    b_eq = np.zeros(horizon)
    b_eq[0] = soc_start
    return a_eq, b_eq


def _cycle_limit(
    spec: BatterySpec, horizon: int, cycle_cap: float | None
) -> tuple[sparse.csr_matrix | None, np.ndarray | None]:
    """Cap total discharged energy at N equivalent full cycles."""
    cycles = cycle_cap if cycle_cap is not None else spec.max_cycles_per_day
    if cycles is None:
        return None, None
    limit = float(cycles) * spec.ac_kwh_per_full_cycle
    row = sparse.csr_matrix(
        (np.ones(horizon), (np.zeros(horizon, dtype=int), np.arange(horizon, 2 * horizon))),
        shape=(1, 3 * horizon),
        dtype=float,
    )
    return row, np.array([limit])


def _terminal_bounds(spec: BatterySpec, soc_start: float, terminal: TerminalMode) -> tuple:
    if terminal == "initial":
        # Pinning the end state stops a run from manufacturing profit by
        # simply finishing empty, which makes days comparable to each other.
        return (soc_start, soc_start)
    if terminal == "atleast":
        return (soc_start, spec.soc_max_kwh)
    if terminal == "free":
        return (spec.soc_min_kwh, spec.soc_max_kwh)
    raise ValueError(f"terminal must be 'initial', 'atleast' or 'free', got {terminal!r}")


def _cap(values: np.ndarray | None, default: float, horizon: int) -> np.ndarray:
    if values is None:
        return np.full(horizon, float(default))
    array = np.asarray(values, dtype=float)
    if array.size != horizon:
        raise ValueError(f"per-slot cap has {array.size} entries, expected {horizon}")
    return np.clip(array, 0.0, None)


def _warn_on_free_money(spec: BatterySpec, buy: np.ndarray, sell: np.ndarray) -> None:
    """Flag slots where charging and discharging at once would pay.

    Round-tripping within a single slot nets
    ``RTE * (sell - wear) - buy`` per kWh charged.  If that is positive the
    LP will happily do it and report an unbounded-looking profit, which is
    always a tariff configuration error rather than a real opportunity.
    """
    margin = spec.round_trip_efficiency * (sell - spec.degradation_cost_jpy_per_kwh) - buy
    offenders = int(np.sum(margin > 1e-9))
    if offenders:
        log.warning(
            "%d slot(s) price a same-slot round trip as profitable (max %.3f JPY/kWh). "
            "The dispatch will exploit them; check the tariff's buy/sell definitions.",
            offenders,
            float(margin.max()),
        )
