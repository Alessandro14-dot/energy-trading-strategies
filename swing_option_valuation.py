"""
Swing Option Valuation — Intrinsic & Extrinsic Value
=====================================================
Models and values gas and power swing options, a cornerstone
instrument in energy supply and portfolio management.

What is a swing option?
    A swing option (also called take-or-pay or flexible supply contract)
    gives the holder the right to vary the quantity taken within bounds:

        Min daily quantity (MDQ) ≤ daily volume ≤ Max daily quantity (MDQ_max)
        Min contract quantity (MCQ) ≤ total period ≤ Max contract quantity (MCQ_max)

    The holder "swings" between the minimum and maximum take,
    optimising based on prevailing spot prices.

    Key parameters:
        Swing rights  : number of times the holder can deviate from base volume
        Nomination    : day-ahead volume nomination
        Penalty cost  : cost of under-delivering against MCQ
        Recall right  : seller's right to recall gas

    Common in:
        - Long-term gas supply agreements (LTSAs)
        - Power purchase agreements (PPAs)
        - LNG supply contracts
        - Virtual power plant (VPP) agreements

Valuation decomposition:
    Total Value = Intrinsic Value + Extrinsic Value

    INTRINSIC VALUE (static, deterministic):
        Value assuming we know future prices today (current forward curve).
        Computed by optimising volume allocation across delivery days:
            IV = Σ max(0, P_i - P_strike) * volume_i
        This is the value locked in by hedging today.

    EXTRINSIC VALUE (optionality, stochastic):
        Additional value from price uncertainty and flexibility to re-optimise.
        EV = Total_Value - Intrinsic_Value
        Captured through Monte Carlo simulation or dynamic programming.

    Rule of thumb (energy markets):
        Extrinsic ≈ 20-50% of total value in normal markets
        Extrinsic → 0 as volatility → 0
        Extrinsic → large in high-vol or spike regimes

Valuation methods:
    1. Intrinsic: forward curve optimisation (deterministic)
    2. Rolling intrinsic: re-optimise daily as forward curve updates
    3. Monte Carlo: simulate price paths, optimise each path
    4. Least-Squares Monte Carlo (Longstaff-Schwartz): regression-based
    5. Dynamic programming: backward induction on price tree

Trading signals from swing options:
    - Intrinsic value spread vs market implied swing value
    - Rolling intrinsic P&L as hedge effectiveness proxy
    - Extrinsic value time decay signal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


# ─── Swing contract specification ────────────────────────────────────────────

@dataclass
class SwingContract:
    """
    Specification of a gas or power swing option contract.

    Parameters
    ----------
    start_date      : delivery period start
    end_date        : delivery period end
    strike_price    : contractual price [€/MWh]
    base_volume_mwh : base daily volume [MWh/day]
    min_daily_pct   : minimum daily take as fraction of base (e.g. 0.80 = 80%)
    max_daily_pct   : maximum daily take as fraction of base (e.g. 1.20 = 120%)
    min_total_pct   : minimum total contract quantity as fraction of base total
    max_total_pct   : maximum total contract quantity as fraction of base total
    swing_rights    : number of times holder can deviate (None = unlimited)
    commodity       : "gas" or "power"
    """
    start_date:       str
    end_date:         str
    strike_price:     float         = 40.0      # €/MWh
    base_volume_mwh:  float         = 100.0     # MWh/day
    min_daily_pct:    float         = 0.80
    max_daily_pct:    float         = 1.20
    min_total_pct:    float         = 0.90
    max_total_pct:    float         = 1.10
    swing_rights:     Optional[int] = None      # None = unlimited
    commodity:        str           = "gas"

    @property
    def delivery_dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start_date, self.end_date, freq="D")

    @property
    def n_days(self) -> int:
        return len(self.delivery_dates)

    @property
    def total_base_volume(self) -> float:
        return self.base_volume_mwh * self.n_days

    @property
    def min_volume_day(self) -> float:
        return self.base_volume_mwh * self.min_daily_pct

    @property
    def max_volume_day(self) -> float:
        return self.base_volume_mwh * self.max_daily_pct

    @property
    def min_total_volume(self) -> float:
        return self.total_base_volume * self.min_total_pct

    @property
    def max_total_volume(self) -> float:
        return self.total_base_volume * self.max_total_pct


# ─── Intrinsic value calculator ──────────────────────────────────────────────

class SwingIntrinsicValuation:
    """
    Computes the intrinsic value of a swing option from the forward curve.

    The intrinsic value is the value assuming we trade optimally
    against today's forward curve, without any uncertainty.

    Optimal strategy:
        - On days where P_i > strike: take maximum volume
        - On days where P_i < strike: take minimum volume
        - Subject to total volume constraints [MCQ_min, MCQ_max]

    When total constraints bind, we use a greedy/LP approach:
        Sort days by (P_i - strike) descending
        Allocate maximum volume to most profitable days
        Until MCQ_max is reached; remaining days get minimum volume

    Parameters
    ----------
    contract     : SwingContract
    forward_prices: pd.Series   Daily forward prices for delivery period [€/MWh]
    """

    def __init__(
        self,
        contract: SwingContract,
        forward_prices: pd.Series,
    ):
        self.contract = contract
        self.fwd      = forward_prices.reindex(contract.delivery_dates)

    def optimal_dispatch(self) -> pd.DataFrame:
        """
        Compute the optimal daily volume allocation (intrinsic dispatch).
        Returns DataFrame with date, forward price, optimal volume, daily P&L.
        """
        ct   = self.contract
        fwd  = self.fwd.values
        n    = ct.n_days
        K    = ct.strike_price
        v_min = ct.min_volume_day
        v_max = ct.max_volume_day
        V_min = ct.min_total_volume
        V_max = ct.max_total_volume

        # Greedy optimal: sort by spread descending, fill max first
        spreads  = fwd - K
        sort_idx = np.argsort(-spreads)   # highest spread first

        volumes  = np.full(n, v_min)      # start at minimum
        total    = v_min * n

        for i in sort_idx:
            if spreads[i] <= 0:
                break   # no benefit to swing up
            add   = v_max - v_min
            if total + add > V_max:
                add = V_max - total
            if add <= 0:
                break
            volumes[i] += add
            total       += add

        # Ensure we meet minimum total volume
        if total < V_min:
            deficit = V_min - total
            for i in sort_idx[::-1]:   # add to worst days first
                add = min(v_max - volumes[i], deficit)
                volumes[i]  += add
                deficit     -= add
                if deficit <= 0:
                    break

        daily_pnl  = (fwd - K) * volumes
        dates      = ct.delivery_dates

        return pd.DataFrame({
            "date":          dates,
            "forward_price": fwd,
            "spread":        fwd - K,
            "volume_mwh":    volumes,
            "daily_pnl":     daily_pnl,
        }).set_index("date")

    def intrinsic_value(self) -> float:
        """Total intrinsic value [€]."""
        dispatch = self.optimal_dispatch()
        return round(float(dispatch["daily_pnl"].sum()), 2)

    def intrinsic_per_mwh(self) -> float:
        """Intrinsic value per MWh of base volume [€/MWh]."""
        iv    = self.intrinsic_value()
        total = self.contract.total_base_volume
        return round(iv / total if total > 0 else 0.0, 4)


# ─── Monte Carlo extrinsic valuation ─────────────────────────────────────────

class SwingMonteCarloValuation:
    """
    Estimates total swing option value via Monte Carlo simulation.
    Extrinsic = MC_Total_Value - Intrinsic_Value

    Price model: Geometric Brownian Motion with mean reversion (Vasicek).
        dP = κ(μ - P)dt + σP dW   (simplified)

    Parameters
    ----------
    contract     : SwingContract
    spot_price   : float   Current spot/prompt price [€/MWh]
    forward_prices: pd.Series  Forward curve for the delivery period
    sigma        : float   Annualised vol [decimal]
    kappa        : float   Mean reversion speed
    n_paths      : int     Number of Monte Carlo paths
    seed         : int
    """

    def __init__(
        self,
        contract: SwingContract,
        spot_price: float,
        forward_prices: pd.Series,
        sigma: float = 0.40,
        kappa: float = 2.0,
        n_paths: int = 2000,
        seed: int = 42,
    ):
        self.contract  = contract
        self.spot      = spot_price
        self.fwd       = forward_prices.reindex(contract.delivery_dates)
        self.sigma     = sigma
        self.kappa     = kappa
        self.n_paths   = n_paths
        self.rng       = np.random.default_rng(seed)

    def simulate_paths(self) -> np.ndarray:
        """
        Simulate price paths for the delivery period.
        Shape: (n_days, n_paths)
        Uses mean-reverting GBM around forward curve.
        """
        n     = self.contract.n_days
        dt    = 1 / 252
        fwd   = self.fwd.fillna(method="ffill").values

        paths = np.zeros((n, self.n_paths))
        paths[0] = self.spot

        for t in range(1, n):
            mu_t = fwd[t]   # forward price as target mean
            z    = self.rng.standard_normal(self.n_paths)
            # Mean-reverting around forward
            drift = self.kappa * (mu_t - paths[t-1]) * dt
            diff  = self.sigma * paths[t-1] * np.sqrt(dt) * z
            paths[t] = np.maximum(paths[t-1] + drift + diff, 1.0)

        return paths

    def value_path(self, prices: np.ndarray) -> float:
        """
        Compute optimal swing value for a single price path.
        Simple greedy dispatch (same as intrinsic but with simulated prices).
        """
        ct    = self.contract
        K     = ct.strike_price
        v_min = ct.min_volume_day
        v_max = ct.max_volume_day
        V_max = ct.max_total_volume
        V_min = ct.min_total_volume

        spreads  = prices - K
        sort_idx = np.argsort(-spreads)
        volumes  = np.full(len(prices), v_min)
        total    = v_min * len(prices)

        for i in sort_idx:
            if spreads[i] <= 0: break
            add = min(v_max - v_min, V_max - total)
            if add <= 0: break
            volumes[i] += add
            total       += add

        if total < V_min:
            deficit = V_min - total
            for i in sort_idx[::-1]:
                add = min(v_max - volumes[i], deficit)
                volumes[i] += add
                deficit    -= add
                if deficit <= 0: break

        return float(np.sum(np.maximum(spreads, 0) * volumes))

    def total_value(self) -> Dict[str, float]:
        """
        Estimate total swing value and decompose into intrinsic + extrinsic.
        """
        paths  = self.simulate_paths()
        values = np.array([self.value_path(paths[:, j]) for j in range(self.n_paths)])

        mc_total    = float(np.mean(values))
        mc_std      = float(np.std(values))
        mc_se       = mc_std / np.sqrt(self.n_paths)

        # Intrinsic from forward curve
        iv_calc  = SwingIntrinsicValuation(self.contract, self.fwd)
        intrinsic = iv_calc.intrinsic_value()
        extrinsic = max(0.0, mc_total - intrinsic)

        base_vol  = self.contract.total_base_volume
        return {
            "intrinsic_value_eur":   round(intrinsic, 2),
            "mc_total_value_eur":    round(mc_total, 2),
            "extrinsic_value_eur":   round(extrinsic, 2),
            "extrinsic_pct":         round(extrinsic / mc_total * 100, 1) if mc_total > 0 else 0.0,
            "value_per_mwh":         round(mc_total / base_vol, 4) if base_vol > 0 else 0.0,
            "intrinsic_per_mwh":     round(intrinsic / base_vol, 4) if base_vol > 0 else 0.0,
            "mc_std_eur":            round(mc_std, 2),
            "mc_95ci_low":           round(mc_total - 1.96 * mc_se, 2),
            "mc_95ci_high":          round(mc_total + 1.96 * mc_se, 2),
            "n_paths":               self.n_paths,
        }

    def plot_paths(self, n_show: int = 50, figsize=(12, 5)) -> plt.Figure:
        """Plot simulated price paths vs forward curve."""
        paths = self.simulate_paths()
        dates = self.contract.delivery_dates
        fig, ax = plt.subplots(figsize=figsize)
        for j in range(min(n_show, self.n_paths)):
            ax.plot(dates, paths[:, j], color="#1565c0", alpha=0.05, lw=0.6)
        ax.plot(dates, self.fwd.values, color="black", lw=1.5, label="Forward Curve")
        ax.axhline(self.contract.strike_price, color="red", lw=1.0, ls="--",
                   label=f"Strike {self.contract.strike_price} €/MWh")
        ax.legend(fontsize=9); ax.set_ylabel("Price [€/MWh]"); ax.set_xlabel("Date")
        ax.set_title(f"Monte Carlo Paths — {self.contract.commodity.upper()} Swing Option "
                     f"({self.n_paths} paths, {n_show} shown)", fontsize=11)
        plt.tight_layout()
        return fig


# ─── Rolling intrinsic strategy ───────────────────────────────────────────────

class RollingIntrinsicStrategy:
    """
    Rolling intrinsic hedge: re-compute and rebalance the intrinsic hedge
    each day as the forward curve updates.

    This captures both intrinsic and a portion of extrinsic value
    through systematic re-optimisation.

    Parameters
    ----------
    contract        : SwingContract
    forward_curve_ts: pd.DataFrame   Columns = delivery dates, index = observation dates
    """

    def __init__(
        self,
        contract: SwingContract,
        forward_curve_ts: pd.DataFrame,
    ):
        self.contract = contract
        self.fwd_ts   = forward_curve_ts
        self.results: Optional[pd.DataFrame] = None

    def run(self) -> pd.DataFrame:
        """
        On each observation date, re-optimise against updated forward curve.
        Track daily change in intrinsic value as rolling hedge P&L.
        """
        dates  = self.fwd_ts.index
        iv_ts  = pd.Series(index=dates, dtype=float)

        for dt in dates:
            fwd_slice = self.fwd_ts.loc[dt]
            iv_calc   = SwingIntrinsicValuation(self.contract, fwd_slice)
            try:
                iv_ts[dt] = iv_calc.intrinsic_value()
            except Exception:
                iv_ts[dt] = np.nan

        daily_pnl = iv_ts.diff()
        self.results = pd.DataFrame({
            "intrinsic_value": iv_ts,
            "daily_pnl":       daily_pnl,
            "cum_pnl":         daily_pnl.cumsum(),
        })
        return self.results

    def plot(self, figsize=(12, 7)) -> plt.Figure:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        df  = self.results
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        fig.suptitle("Rolling Intrinsic Hedge — Swing Option", fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["intrinsic_value"], color="#1565c0", lw=1.0)
        ax.set_ylabel("€"); ax.set_title("Intrinsic Value (€)", fontsize=10)

        ax = axes[1]
        ax.fill_between(df.index, df["cum_pnl"], 0,
                        where=df["cum_pnl"] >= 0, color="green", alpha=0.4)
        ax.fill_between(df.index, df["cum_pnl"], 0,
                        where=df["cum_pnl"] < 0,  color="red",   alpha=0.4)
        ax.plot(df.index, df["cum_pnl"], color="black", lw=0.8)
        ax.set_ylabel("€ cumul."); ax.set_title("Rolling Intrinsic Cumulative P&L", fontsize=10)
        ax.set_xlabel("Date")

        plt.tight_layout()
        return fig


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Contract spec
    contract = SwingContract(
        start_date      = "2024-10-01",
        end_date        = "2025-03-31",
        strike_price    = 45.0,
        base_volume_mwh = 500.0,
        min_daily_pct   = 0.80,
        max_daily_pct   = 1.20,
        min_total_pct   = 0.90,
        max_total_pct   = 1.10,
        commodity       = "gas",
    )
    print(f"\nContract: {contract.commodity.upper()} Swing")
    print(f"  Delivery:     {contract.start_date} → {contract.end_date}  ({contract.n_days} days)")
    print(f"  Strike:       {contract.strike_price} €/MWh")
    print(f"  Base volume:  {contract.base_volume_mwh} MWh/day  ({contract.total_base_volume:,.0f} MWh total)")
    print(f"  Daily range:  [{contract.min_volume_day:.0f}, {contract.max_volume_day:.0f}] MWh")

    # Forward curve (seasonal: higher in winter)
    dates  = contract.delivery_dates
    doy    = np.array([d.dayofyear for d in dates])
    fwd    = 48 + 10*np.cos(2*np.pi*(doy-15)/365) + rng.normal(0, 2, len(dates))
    fwd_s  = pd.Series(fwd, index=dates)

    # 1) Intrinsic valuation
    iv_calc = SwingIntrinsicValuation(contract, fwd_s)
    dispatch = iv_calc.optimal_dispatch()
    iv       = iv_calc.intrinsic_value()
    iv_mwh   = iv_calc.intrinsic_per_mwh()

    print(f"\n=== Intrinsic Valuation ===")
    print(f"  Intrinsic value:    €{iv:,.2f}")
    print(f"  Per MWh (base):     {iv_mwh:.4f} €/MWh")
    print(f"\n  Dispatch summary (first 10 days):")
    print(dispatch.head(10).round(2).to_string())

    # 2) Monte Carlo total value
    mc = SwingMonteCarloValuation(
        contract=contract, spot_price=50.0,
        forward_prices=fwd_s, sigma=0.45, kappa=2.0, n_paths=1000,
    )
    val = mc.total_value()
    print(f"\n=== Monte Carlo Valuation ({val['n_paths']} paths) ===")
    for k, v in val.items():
        print(f"  {k:30s}: {v}")

    fig1 = mc.plot_paths(n_show=80)
    fig1.savefig("swing_mc_paths.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → swing_mc_paths.png")
