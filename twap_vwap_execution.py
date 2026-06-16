"""
TWAP & VWAP Execution Algorithms — Energy Markets
===================================================
Implements Time-Weighted Average Price (TWAP) and Volume-Weighted
Average Price (VWAP) execution strategies adapted for energy markets.

Context:
    In energy markets, large orders cannot be executed all at once
    without significant market impact. Systematic slicing is essential.

    Key differences from equity markets:
        - Energy products have a SHAPE (delivery profile varies by hour)
        - Liquidity concentrated around certain times (DA auction, peak hours)
        - Gate closure creates urgency near delivery
        - Cross-product hedging (power + gas + EUA simultaneously)
        - Seasonal volume profiles (winter vs summer)

    TWAP (Time-Weighted Average Price):
        - Slice order equally across time
        - Simple, predictable, minimal information leakage
        - Best when: no strong intraday pattern, execution time is long
        - Risk: ignores volume/liquidity timing

    VWAP (Volume-Weighted Average Price):
        - Slice proportional to expected intraday volume profile
        - Participates more when market is liquid
        - Best when: intraday volume is predictable (seasonal shape)
        - Benchmark: VWAP execution means you get the market average

    Energy-specific considerations:
        - EPEX SPOT volume profile: peaks at DA auction (12:00) and near gate closure
        - Gas: TTF volume peaks at 09:00-11:00 CET
        - Weather-driven volatility clusters: avoid large executions during high-vol
        - Block vs hourly products: different liquidity profiles

Slippage and market impact:
    Market impact ≈ σ * sqrt(Q / ADV)
    where Q = order size, ADV = average daily volume

    Energy markets: typically thin, so impact matters even for moderate sizes.
    Rule of thumb: limit each child order to < 5-10% of period liquidity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Literal, Tuple
from enum import Enum


# ─── Enums and types ──────────────────────────────────────────────────────────

class Side(str, Enum):
    BUY  = "buy"
    SELL = "sell"

class AlgoType(str, Enum):
    TWAP = "TWAP"
    VWAP = "VWAP"
    POV  = "POV"    # Percentage of Volume


# ─── Order specification ──────────────────────────────────────────────────────

@dataclass
class ParentOrder:
    """Large order to be executed algorithmically."""
    order_id: str
    side: Side
    total_volume_mw: float           # total MW to execute
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    product: str                     # e.g. "Power_DE_H8", "TTF_M+1"
    limit_price: Optional[float]     # None = market order
    algo: AlgoType = AlgoType.TWAP
    urgency: float = 0.5             # 0 = passive, 1 = aggressive


@dataclass
class ChildOrder:
    """Single slice of a parent order."""
    parent_id: str
    slice_num: int
    side: Side
    volume_mw: float
    target_time: pd.Timestamp
    limit_price: Optional[float]
    executed_volume: float = 0.0
    executed_price: float  = 0.0
    slippage: float        = 0.0


# ─── Volume profile (VWAP curve) ─────────────────────────────────────────────

@dataclass
class VolumeProfile:
    """
    Intraday volume profile for VWAP benchmark calculation.
    Based on historical trading volumes by time bucket.
    """
    market: str = "EPEX_DE"
    bucket_minutes: int = 15

    def get_profile(self) -> pd.Series:
        """
        Return normalised volume profile (sums to 1.0) by 15-min bucket.
        Based on stylised facts of EPEX SPOT continuous trading.
        """
        # 96 buckets (24h × 4 per hour)
        buckets = np.arange(96)
        # Stylised EPEX profile:
        #   - Peak at 08:00-09:00 (market open, DA published)
        #   - Secondary peak at 14:00-16:00 (new information)
        #   - Drop near midnight
        profile = np.ones(96) * 0.005   # base
        # Morning ramp
        profile[32:40] += np.linspace(0, 0.025, 8)   # 08:00-10:00
        profile[40:44] += 0.025                        # 10:00-11:00
        # Midday peak (DA publication zone)
        profile[44:52] += np.linspace(0.025, 0.035, 8)# 11:00-13:00
        profile[52:56] += 0.030                        # 13:00-14:00
        # Afternoon secondary
        profile[56:64] += np.linspace(0.030, 0.020, 8)# 14:00-16:00
        profile[64:72] += 0.015                        # 16:00-18:00
        # Evening
        profile[72:80] += 0.008
        # Normalise
        profile = profile / profile.sum()
        idx = pd.timedelta_range("0h", periods=96, freq="15min")
        return pd.Series(profile, index=idx, name="volume_fraction")

    def get_daily_profile(self, date: pd.Timestamp) -> pd.Series:
        """Return profile anchored to a specific date."""
        base  = self.get_profile()
        times = pd.date_range(date.normalize(), periods=96, freq="15min")
        return pd.Series(base.values, index=times, name="volume_fraction")


# ─── TWAP algorithm ───────────────────────────────────────────────────────────

class TWAPAlgorithm:
    """
    Time-Weighted Average Price execution algorithm.

    Slices the parent order into equal-sized child orders
    distributed uniformly across the execution window.

    Parameters
    ----------
    order       : ParentOrder
    n_slices    : int            Number of equal time slices
    randomise   : bool           Add ±10% random jitter to avoid fingerprinting
    """

    def __init__(
        self,
        order: ParentOrder,
        n_slices: int = 10,
        randomise: bool = True,
        seed: int = 42,
    ):
        self.order      = order
        self.n_slices   = n_slices
        self.randomise  = randomise
        self.rng        = np.random.default_rng(seed)

    def generate_schedule(self) -> List[ChildOrder]:
        """Generate list of child orders with equal time spacing."""
        start   = self.order.start_time
        end     = self.order.end_time
        total   = self.order.total_volume_mw
        n       = self.n_slices
        base_vol = total / n

        interval = (end - start) / n
        children = []

        for i in range(n):
            # Jitter ±10% of base volume
            if self.randomise:
                vol = base_vol * (1 + self.rng.uniform(-0.10, 0.10))
            else:
                vol = base_vol

            t = start + interval * i + interval * self.rng.uniform(0, 0.5)
            children.append(ChildOrder(
                parent_id   = self.order.order_id,
                slice_num   = i,
                side        = self.order.side,
                volume_mw   = round(vol, 3),
                target_time = t,
                limit_price = self.order.limit_price,
            ))

        return children

    def target_volumes(self) -> pd.Series:
        """Return target volume schedule as time series."""
        schedule = self.generate_schedule()
        times    = [c.target_time for c in schedule]
        vols     = [c.volume_mw   for c in schedule]
        return pd.Series(vols, index=times, name="target_volume_mw")


# ─── VWAP algorithm ───────────────────────────────────────────────────────────

class VWAPAlgorithm:
    """
    Volume-Weighted Average Price execution algorithm.

    Slices the parent order proportional to the expected intraday
    volume profile (liquidity participation).

    Parameters
    ----------
    order          : ParentOrder
    volume_profile : VolumeProfile
    participation  : float         Target participation rate [0-1]
    """

    def __init__(
        self,
        order: ParentOrder,
        volume_profile: Optional[VolumeProfile] = None,
        participation: float = 0.05,    # 5% of expected volume
        seed: int = 42,
    ):
        self.order       = order
        self.vol_profile = volume_profile or VolumeProfile()
        self.particip    = participation
        self.rng         = np.random.default_rng(seed)

    def generate_schedule(self) -> List[ChildOrder]:
        """
        Generate child orders proportional to expected volume profile.
        """
        start   = self.order.start_time
        end     = self.order.end_time
        total   = self.order.total_volume_mw
        profile = self.vol_profile.get_daily_profile(start)

        # Filter profile to execution window
        mask    = (profile.index >= start) & (profile.index <= end)
        window  = profile[mask]
        if window.sum() == 0 or len(window) == 0:
            # Fallback to TWAP if no volume profile in window
            window  = pd.Series(np.ones(10), index=pd.date_range(start, end, periods=10))

        # Normalise within window
        weights = window / window.sum()
        children = []

        for i, (t, w) in enumerate(weights.items()):
            vol = total * w
            if vol < 0.01:
                continue
            jitter = self.rng.uniform(0, (window.index[1] - window.index[0]).seconds) if len(window) > 1 else 0
            children.append(ChildOrder(
                parent_id   = self.order.order_id,
                slice_num   = i,
                side        = self.order.side,
                volume_mw   = round(float(vol), 3),
                target_time = t + pd.Timedelta(seconds=jitter),
                limit_price = self.order.limit_price,
            ))

        return children

    def vwap_benchmark(self, market_prices: pd.Series, market_volumes: pd.Series) -> float:
        """
        Compute actual market VWAP over execution window as benchmark.
        """
        start = self.order.start_time
        end   = self.order.end_time
        mask  = (market_prices.index >= start) & (market_prices.index <= end)
        p     = market_prices[mask]
        v     = market_volumes[mask]
        if v.sum() == 0:
            return float(p.mean()) if len(p) > 0 else np.nan
        return float((p * v).sum() / v.sum())


# ─── Execution simulator ──────────────────────────────────────────────────────

class ExecutionSimulator:
    """
    Simulates execution of a schedule against synthetic market data.

    Models:
        - Bid-ask spread impact
        - Market impact (temporary + permanent)
        - Fill probability (partial fills near limit price)

    Parameters
    ----------
    schedule    : list of ChildOrder
    market_prices: pd.Series   Mid-price time series [€/MWh]
    bid_ask_spread: float      Half spread [€/MWh]
    impact_coeff: float        Market impact coefficient (Kyle's lambda proxy)
    """

    def __init__(
        self,
        schedule: List[ChildOrder],
        market_prices: pd.Series,
        bid_ask_spread: float = 0.25,
        impact_coeff: float   = 0.02,
        seed: int = 42,
    ):
        self.schedule   = schedule
        self.prices     = market_prices
        self.spread     = bid_ask_spread
        self.impact_k   = impact_coeff
        self.rng        = np.random.default_rng(seed)

    def simulate(self) -> pd.DataFrame:
        """Execute schedule and return execution report."""
        records = []
        for child in self.schedule:
            t = child.target_time
            # Get nearest market price
            if len(self.prices) == 0:
                continue
            idx   = self.prices.index.get_indexer([t], method="nearest")[0]
            mid   = float(self.prices.iloc[idx])

            # Bid-ask cost
            side_sign = 1 if child.side == Side.BUY else -1
            ba_cost   = self.spread * side_sign

            # Market impact (temporary, proportional to sqrt of volume)
            impact    = self.impact_k * np.sqrt(child.volume_mw) * side_sign

            exec_price = mid + ba_cost + impact + self.rng.normal(0, 0.05)

            # Limit price check
            filled = True
            if child.limit_price is not None:
                if child.side == Side.BUY  and exec_price > child.limit_price:
                    filled = False
                if child.side == Side.SELL and exec_price < child.limit_price:
                    filled = False

            slippage = (exec_price - mid) * side_sign if filled else 0.0

            records.append({
                "slice_num":      child.slice_num,
                "target_time":    child.target_time,
                "volume_mw":      child.volume_mw if filled else 0.0,
                "mid_price":      mid,
                "exec_price":     exec_price if filled else np.nan,
                "slippage":       slippage if filled else np.nan,
                "filled":         filled,
                "ba_cost":        ba_cost if filled else 0.0,
                "impact_cost":    impact   if filled else 0.0,
            })

        return pd.DataFrame(records)

    def execution_quality(self, report: pd.DataFrame, benchmark_price: Optional[float] = None) -> dict:
        """Compute execution quality metrics."""
        filled   = report[report["filled"]]
        if len(filled) == 0:
            return {"fill_rate": 0.0}

        total_vol   = filled["volume_mw"].sum()
        vwap_exec   = (filled["exec_price"] * filled["volume_mw"]).sum() / total_vol if total_vol > 0 else np.nan
        avg_slip    = filled["slippage"].mean()
        total_impact= filled["impact_cost"].mean()
        total_ba    = filled["ba_cost"].mean()

        result = {
            "fill_rate_pct":      round(filled["volume_mw"].sum() / report["volume_mw"].sum() * 100, 1),
            "total_volume_mw":    round(total_vol, 2),
            "vwap_executed":      round(vwap_exec, 4) if not np.isnan(vwap_exec) else None,
            "avg_slippage":       round(avg_slip, 4),
            "avg_ba_cost":        round(total_ba, 4),
            "avg_impact_cost":    round(total_impact, 4),
            "total_cost_eur_mwh": round(avg_slip, 4),
        }
        if benchmark_price is not None:
            result["vs_benchmark"] = round(vwap_exec - benchmark_price, 4)
        return result


# ─── Full execution workflow ──────────────────────────────────────────────────

def run_execution_comparison(
    total_mw: float      = 100.0,
    n_slices: int        = 20,
    participation: float = 0.05,
    seed: int            = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Run TWAP vs VWAP comparison on a synthetic parent order.
    Returns dict with execution reports for each algorithm.
    """
    rng    = np.random.default_rng(seed)
    start  = pd.Timestamp("2024-01-15 08:00")
    end    = pd.Timestamp("2024-01-15 14:00")

    # Synthetic market prices
    n_min  = int((end - start).total_seconds() / 60) + 1
    times  = pd.date_range(start, periods=n_min, freq="min")
    mid    = 80.0 + np.cumsum(rng.normal(0, 0.2, n_min))
    market = pd.Series(mid, index=times)

    order = ParentOrder(
        order_id = "ORD001",
        side     = Side.BUY,
        total_volume_mw = total_mw,
        start_time = start,
        end_time   = end,
        product    = "Power_DE_H8-H14",
        limit_price = None,
        algo        = AlgoType.TWAP,
    )

    results = {}

    # TWAP
    twap_algo     = TWAPAlgorithm(order, n_slices=n_slices, randomise=True, seed=seed)
    twap_schedule = twap_algo.generate_schedule()
    twap_sim      = ExecutionSimulator(twap_schedule, market, bid_ask_spread=0.20, impact_coeff=0.015)
    twap_report   = twap_sim.simulate()
    results["TWAP"] = twap_report

    # VWAP
    vwap_algo     = VWAPAlgorithm(order, participation=participation, seed=seed)
    vwap_schedule = vwap_algo.generate_schedule()
    vwap_sim      = ExecutionSimulator(vwap_schedule, market, bid_ask_spread=0.20, impact_coeff=0.015)
    vwap_report   = vwap_sim.simulate()
    results["VWAP"] = vwap_report

    return results, market, order


def plot_execution_comparison(
    reports: Dict[str, pd.DataFrame],
    market: pd.Series,
    figsize=(14, 10),
) -> plt.Figure:
    """Plot TWAP vs VWAP execution comparison."""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=False)
    fig.suptitle("TWAP vs VWAP Execution Comparison", fontsize=13, fontweight="bold")

    colours = {"TWAP": "#1565c0", "VWAP": "#c62828"}

    ax = axes[0]
    ax.plot(market.index, market.values, color="#333", lw=0.8, label="Market Mid")
    for algo, rep in reports.items():
        filled = rep[rep["filled"]]
        ax.scatter(filled["target_time"], filled["exec_price"],
                   label=f"{algo} Fills", color=colours[algo], s=30, zorder=5, alpha=0.8)
    ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("Execution Prices vs Market", fontsize=10)

    ax = axes[1]
    for algo, rep in reports.items():
        ax.bar(rep["slice_num"] + (0.4 if algo == "VWAP" else 0),
               rep["volume_mw"], width=0.35, label=algo, color=colours[algo], alpha=0.7)
    ax.legend(fontsize=8); ax.set_ylabel("MW/slice"); ax.set_title("Volume Schedule by Slice", fontsize=10)
    ax.set_xlabel("Slice #")

    ax = axes[2]
    for algo, rep in reports.items():
        filled = rep[rep["filled"]]
        ax.bar(rep["slice_num"] + (0.4 if algo == "VWAP" else 0),
               rep["slippage"].fillna(0), width=0.35, label=algo, color=colours[algo], alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("Slippage by Slice", fontsize=10)
    ax.set_xlabel("Slice #")

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== TWAP vs VWAP Execution Comparison ===\n")
    reports, market, order = run_execution_comparison(
        total_mw=100.0, n_slices=20, participation=0.05
    )

    sim_twap = ExecutionSimulator([], market)
    for algo, rep in reports.items():
        filled = rep[rep["filled"]]
        total_vol = filled["volume_mw"].sum()
        vwap_ex   = (filled["exec_price"] * filled["volume_mw"]).sum() / total_vol if total_vol > 0 else np.nan
        print(f"  {algo}:")
        print(f"    Fill rate:      {filled['volume_mw'].sum() / rep['volume_mw'].sum() * 100:.1f}%")
        print(f"    VWAP executed:  {vwap_ex:.3f} €/MWh")
        print(f"    Avg slippage:   {filled['slippage'].mean():.4f} €/MWh")
        print(f"    Total volume:   {total_vol:.1f} MW\n")

    # Volume profile
    vp = VolumeProfile(market="EPEX_DE")
    profile = vp.get_profile()
    print(f"Peak volume bucket: {profile.idxmax()} ({profile.max()*100:.2f}% of daily volume)")

    fig = plot_execution_comparison(reports, market)
    fig.savefig("twap_vwap_comparison.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → twap_vwap_comparison.png")
