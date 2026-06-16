"""
Merit Order Model — Stack-Based Power Price Simulation
=======================================================
Simulates the electricity price by building the supply stack
(merit order curve) from marginal costs of each generation technology.

Theory:
    In a competitive electricity market, the price is set by the
    marginal cost of the last unit dispatched to meet demand.
    This is the MERIT ORDER: plants are dispatched cheapest first.

    Merit order (typical European, low to high marginal cost):
        1. Nuclear          ~  5–15  €/MWh  (must-run / low VC)
        2. Run-of-river     ~  0–5   €/MWh
        3. Wind / Solar     ~  0     €/MWh  (zero fuel cost)
        4. Lignite/Brown    ~ 15–35  €/MWh
        5. Hard Coal (clean)~ 35–60  €/MWh
        6. CCGT (clean)     ~ 50–120 €/MWh
        7. OCGT / Oil       ~ 80–200 €/MWh
        8. Demand response  ~ 200+   €/MWh

    The MARGINAL PLANT sets the price for all infra-marginal units.

    Key formula (CCGT marginal cost):
        MC_CCGT = Gas_price / efficiency + EUA * CO2_factor_el + O&M

Strategy:
    1. Build merit order from fuel prices and capacities
    2. Compute model price by finding marginal plant at given demand
    3. Trade when actual DA price deviates significantly from model price
    4. Model price is the "fair value" — deviations are mean-reverting

Use cases:
    - Day-ahead price forecasting
    - Identifying over/underpriced hours
    - Fuel switch price computation
    - Capacity value estimation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ─── Generation technology dataclass ─────────────────────────────────────────

@dataclass
class GenerationUnit:
    """A single technology block in the merit order."""
    name: str
    capacity_mw: float
    fuel_type: str                  # "nuclear","coal","gas","oil","wind","solar","hydro"
    base_vc_eur_mwh: float          # base variable cost (fuel-independent part)
    fuel_consumption: float = 0.0   # MWh_fuel / MWh_el (= 1/efficiency, 0 for renewables)
    co2_factor_el: float = 0.0      # t CO2 / MWh_el (0 for non-combustion)
    must_run: bool = False          # True = always dispatched (nuclear, RoR)
    colour: str = "#888888"         # for plotting


# ─── Default European merit order ────────────────────────────────────────────

def default_merit_order() -> List[GenerationUnit]:
    return [
        GenerationUnit("Nuclear",        capacity_mw=10_000, fuel_type="nuclear",
                       base_vc_eur_mwh=10.0,  must_run=True,  colour="#7e57c2"),
        GenerationUnit("Run-of-River",   capacity_mw= 5_000, fuel_type="hydro",
                       base_vc_eur_mwh= 2.0,  must_run=True,  colour="#29b6f6"),
        GenerationUnit("Wind",           capacity_mw=25_000, fuel_type="wind",
                       base_vc_eur_mwh= 0.0,  must_run=False, colour="#a5d6a7"),
        GenerationUnit("Solar PV",       capacity_mw=15_000, fuel_type="solar",
                       base_vc_eur_mwh= 0.0,  must_run=False, colour="#fff176"),
        GenerationUnit("Lignite",        capacity_mw=10_000, fuel_type="lignite",
                       base_vc_eur_mwh= 5.0,  fuel_consumption=2.78,
                       co2_factor_el=0.98,    colour="#8d6e63"),
        GenerationUnit("Hard Coal",      capacity_mw= 8_000, fuel_type="coal",
                       base_vc_eur_mwh= 3.0,  fuel_consumption=2.70,
                       co2_factor_el=0.91,    colour="#455a64"),
        GenerationUnit("CCGT",           capacity_mw=12_000, fuel_type="gas",
                       base_vc_eur_mwh= 2.0,  fuel_consumption=2.00,
                       co2_factor_el=0.74,    colour="#ff8f00"),
        GenerationUnit("OCGT",           capacity_mw= 4_000, fuel_type="gas",
                       base_vc_eur_mwh= 3.5,  fuel_consumption=2.86,
                       co2_factor_el=1.06,    colour="#e65100"),
        GenerationUnit("Oil / Diesel",   capacity_mw= 1_500, fuel_type="oil",
                       base_vc_eur_mwh= 5.0,  fuel_consumption=2.50,
                       co2_factor_el=0.85,    colour="#b71c1c"),
    ]


# ─── Merit order model ────────────────────────────────────────────────────────

class MeritOrderModel:
    """
    Builds the merit order supply stack and computes the model power price.

    Parameters
    ----------
    units        : list of GenerationUnit
    gas_price    : float or pd.Series   Gas price [€/MWh].
    coal_price   : float or pd.Series   Coal price [€/MWh_th].
    eua_price    : float or pd.Series   EUA price [€/t CO2].
    lignite_price: float                Lignite price [€/MWh_th] (often fixed).
    oil_price    : float or pd.Series   Oil price [€/MWh_th].
    """

    FUEL_MAP = {
        "gas":     "gas_price",
        "coal":    "coal_price",
        "lignite": "lignite_price",
        "oil":     "oil_price",
    }

    def __init__(
        self,
        units: Optional[List[GenerationUnit]] = None,
        gas_price: float = 40.0,
        coal_price: float = 10.0,
        eua_price: float = 65.0,
        lignite_price: float = 3.5,
        oil_price: float = 60.0,
    ):
        self.units         = units or default_merit_order()
        self.gas_price     = gas_price
        self.coal_price    = coal_price
        self.eua_price     = eua_price
        self.lignite_price = lignite_price
        self.oil_price     = oil_price

    def marginal_cost(self, unit: GenerationUnit) -> float:
        """
        Compute variable cost [€/MWh_el] for a single unit.
        MC = base_vc + fuel_price * fuel_consumption + EUA * CO2_factor_el
        """
        fuel_prices = {
            "gas_price":     self.gas_price,
            "coal_price":    self.coal_price,
            "lignite_price": self.lignite_price,
            "oil_price":     self.oil_price,
        }
        fuel_cost = 0.0
        fp_key = self.FUEL_MAP.get(unit.fuel_type, None)
        if fp_key and unit.fuel_consumption > 0:
            fuel_cost = fuel_prices[fp_key] * unit.fuel_consumption

        carbon_cost = self.eua_price * unit.co2_factor_el
        return unit.base_vc_eur_mwh + fuel_cost + carbon_cost

    def build_stack(self) -> pd.DataFrame:
        """
        Build the merit order supply stack.
        Returns DataFrame sorted by marginal cost (cheapest first).
        """
        rows = []
        cumulative_mw = 0.0
        for u in self.units:
            mc = self.marginal_cost(u)
            rows.append({
                "name":           u.name,
                "fuel_type":      u.fuel_type,
                "capacity_mw":    u.capacity_mw,
                "marginal_cost":  round(mc, 2),
                "must_run":       u.must_run,
                "colour":         u.colour,
                "cum_capacity_start": cumulative_mw,
            })
            cumulative_mw += u.capacity_mw
        df = pd.DataFrame(rows).sort_values("marginal_cost").reset_index(drop=True)
        # Recompute cumulative capacity after sorting
        df["cum_capacity_start"] = df["capacity_mw"].cumsum().shift(1).fillna(0)
        df["cum_capacity_end"]   = df["capacity_mw"].cumsum()
        return df

    def model_price(self, demand_mw: float) -> float:
        """
        Find the marginal plant at a given demand level.
        Returns the marginal cost of the dispatched plant [€/MWh].
        """
        stack = self.build_stack()
        for _, row in stack.iterrows():
            if row["cum_capacity_end"] >= demand_mw:
                return row["marginal_cost"]
        # Demand exceeds all available capacity → scarcity price
        return 3000.0

    def price_series(self, demand_series: pd.Series) -> pd.Series:
        """Compute model price for a time series of demand values."""
        prices = demand_series.apply(self.model_price)
        prices.name = "merit_order_price"
        return prices

    def fuel_switch_price(
        self,
        from_fuel: str = "coal",
        to_fuel: str   = "gas",
    ) -> float:
        """
        EUA price at which 'to_fuel' technology undercuts 'from_fuel'.
        Solves: MC_from = MC_to for EUA.
        """
        # Get representative units
        from_unit = next((u for u in self.units if u.fuel_type == from_fuel), None)
        to_unit   = next((u for u in self.units if u.fuel_type == to_fuel),   None)
        if from_unit is None or to_unit is None:
            return np.nan

        # MC_from + EUA*(co2_from - co2_to) = MC_to (at switch price)
        mc_from_no_co2 = (self.marginal_cost(from_unit) -
                          self.eua_price * from_unit.co2_factor_el)
        mc_to_no_co2   = (self.marginal_cost(to_unit) -
                          self.eua_price * to_unit.co2_factor_el)
        delta_co2      = from_unit.co2_factor_el - to_unit.co2_factor_el
        if abs(delta_co2) < 1e-6:
            return np.nan
        switch_eua = (mc_from_no_co2 - mc_to_no_co2) / delta_co2
        return round(switch_eua, 2)

    def plot_stack(self, demand_mw: Optional[float] = None,
                   figsize=(12, 6)) -> plt.Figure:
        """Plot the merit order supply stack (supply curve)."""
        stack = self.build_stack()
        fig, ax = plt.subplots(figsize=figsize)
        for _, row in stack.iterrows():
            ax.barh(
                y=row["marginal_cost"],
                width=row["capacity_mw"],
                left=row["cum_capacity_start"],
                height=max(row["marginal_cost"] * 0.08, 1.5),
                color=row["colour"],
                alpha=0.85,
                label=row["name"],
            )
            ax.text(
                row["cum_capacity_start"] + row["capacity_mw"] / 2,
                row["marginal_cost"] + max(row["marginal_cost"] * 0.04, 0.8),
                row["name"], ha="center", va="bottom", fontsize=7, rotation=45
            )
        if demand_mw is not None:
            mp = self.model_price(demand_mw)
            ax.axvline(demand_mw, color="red",  lw=1.5, ls="--", label=f"Demand {demand_mw/1000:.0f} GW")
            ax.axhline(mp,        color="black", lw=1.2, ls="--", label=f"Model price {mp:.1f} €/MWh")

        ax.set_xlabel("Cumulative Capacity [MW]")
        ax.set_ylabel("Marginal Cost [€/MWh]")
        ax.set_title(f"Merit Order Stack  |  Gas: {self.gas_price}€/MWh  "
                     f"Coal: {self.coal_price}€/MWh_th  EUA: {self.eua_price}€/t",
                     fontsize=11)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper left")
        plt.tight_layout()
        return fig


# ─── Strategy using merit order model ────────────────────────────────────────

class MeritOrderStrategy:
    """
    Trade when actual DA price deviates from the merit order model price.

    Parameters
    ----------
    actual_prices : pd.Series        Actual DA prices [€/MWh].
    demand        : pd.Series        System demand [MW].
    gas_prices    : pd.Series        Gas prices [€/MWh].
    coal_prices   : pd.Series        Coal prices [€/MWh_th].
    eua_prices    : pd.Series        EUA prices [€/t].
    lookback      : int              Rolling window for z-score.
    entry_zscore  : float
    exit_zscore   : float
    """

    def __init__(
        self,
        actual_prices: pd.Series,
        demand: pd.Series,
        gas_prices: pd.Series,
        coal_prices: pd.Series,
        eua_prices: pd.Series,
        lookback: int = 20,
        entry_zscore: float = 1.6,
        exit_zscore: float  = 0.4,
        stop_zscore: float  = 3.5,
    ):
        self.prices      = actual_prices
        self.demand      = demand
        self.gas         = gas_prices
        self.coal        = coal_prices
        self.eua         = eua_prices
        self.lookback    = lookback
        self.entry_z     = entry_zscore
        self.exit_z      = exit_zscore
        self.stop_z      = stop_zscore
        self.results: Optional[pd.DataFrame] = None

    def compute_model_prices(self) -> pd.Series:
        """Compute daily merit order model price using daily fuel inputs."""
        model_px = pd.Series(np.nan, index=self.prices.index, name="model_price")
        for dt in self.prices.index:
            mdl = MeritOrderModel(
                gas_price  = float(self.gas.get(dt, 40)),
                coal_price = float(self.coal.get(dt, 10)),
                eua_price  = float(self.eua.get(dt, 65)),
            )
            model_px[dt] = mdl.model_price(float(self.demand.get(dt, 45000)))
        return model_px

    def run(self) -> pd.DataFrame:
        model_px   = self.compute_model_prices()
        deviation  = self.prices - model_px
        roll_std   = deviation.rolling(self.lookback).std()
        zscore     = deviation / roll_std.replace(0, np.nan)

        position = pd.Series(0.0, index=self.prices.index)
        current  = 0
        for i in range(self.lookback, len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z): continue
            if current == 0:
                if z < -self.entry_z:   current =  1   # price cheap vs model → buy
                elif z > self.entry_z:  current = -1   # price rich vs model  → sell
            elif current ==  1:
                if z >= -self.exit_z or z <= -self.stop_z: current = 0
            elif current == -1:
                if z <=  self.exit_z or z >=  self.stop_z: current = 0
            position.iloc[i] = current

        daily_pnl = position.shift(1) * self.prices.diff()
        self.results = pd.DataFrame({
            "actual_price": self.prices,
            "model_price":  model_px,
            "deviation":    deviation,
            "zscore":       zscore,
            "position":     position,
            "daily_pnl":    daily_pnl,
            "cum_pnl":      daily_pnl.cumsum(),
        })
        return self.results

    def summary(self) -> dict:
        if self.results is None: raise RuntimeError("Call run() first.")
        pnl    = self.results["daily_pnl"].dropna()
        dev    = self.results["deviation"].dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        return {
            "avg_model_deviation": round(dev.mean(), 2),
            "std_model_deviation": round(dev.std(), 2),
            "total_pnl":           round(pnl.sum(), 2),
            "sharpe_ratio":        round(sharpe, 3),
            "max_drawdown":        round((pnl.cumsum() - pnl.cumsum().cummax()).min(), 2),
            "win_rate":            round((pnl > 0).mean(), 3),
        }


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Static stack plot
    mdl = MeritOrderModel(gas_price=45, coal_price=11, eua_price=70)
    print(f"\nFuel switch EUA price (coal→gas): {mdl.fuel_switch_price('coal','gas')} €/t")
    print("\nMerit Order Stack:")
    print(mdl.build_stack()[["name","marginal_cost","capacity_mw"]].to_string(index=False))
    fig = mdl.plot_stack(demand_mw=48_000)
    fig.savefig("merit_order_stack.png", dpi=150, bbox_inches="tight")
    print("\nStack chart saved → merit_order_stack.png")

    # 2) Time-series strategy
    rng   = np.random.default_rng(42)
    n     = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    gas   = pd.Series(np.clip(40 + np.cumsum(rng.normal(0, 1.5, n)), 15, 200), index=dates)
    coal  = pd.Series(np.clip(10 + np.cumsum(rng.normal(0, 0.3, n)),  5,  40), index=dates)
    eua   = pd.Series(np.clip(65 + np.cumsum(rng.normal(0, 0.8, n)), 20, 130), index=dates)
    dem   = pd.Series(np.clip(45000 + rng.normal(0, 3000, n), 30000, 65000),   index=dates)

    # Build model prices and add noise for actual
    actual = pd.Series([
        MeritOrderModel(gas_price=gas[i], coal_price=coal[i], eua_price=eua[i])
        .model_price(dem.iloc[i]) + rng.normal(0, 6)
        for i in range(n)
    ], index=dates, name="actual_price")

    strat = MeritOrderStrategy(actual, dem, gas, coal, eua)
    strat.run()
    stats = strat.summary()
    print("\n=== Merit Order Strategy — Summary ===")
    for k, v in stats.items():
        print(f"  {k:28s}: {v}")
