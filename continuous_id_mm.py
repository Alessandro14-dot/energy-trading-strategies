"""
Continuous Intraday Market Making Strategy
==========================================
Implements a market-making strategy for European continuous intraday
power markets (EPEX SPOT continuous, GME MI continuous).

Market structure:
    After the day-ahead auction, continuous intraday trading opens.
    Products traded: hourly, half-hourly (15-min in DE), and block contracts.

    Key characteristics:
        - Order-book based (limit orders, market orders)
        - Gate closure: typically H-1 (1 hour before delivery)
        - Tick size: 0.01 €/MWh
        - Lot size: 0.1 MW
        - Cross-border coupling: SIDC (Single Intraday Coupling)

    Market making role:
        - Post bid (buy) and ask (sell) orders simultaneously
        - Earn the bid-ask spread
        - Manage inventory risk (delta exposure)
        - Hedge residual position via DA or other instruments

Market making P&L:
    Revenue = Spread earned per round-trip * volume traded
    Cost    = Inventory risk (adverse selection) + hedging cost

    Spread = ask_price - bid_price
    In energy ID markets, typical spread: 0.10 - 2.00 €/MWh
    (varies by product, time to delivery, market liquidity)

Strategy components:
    1. QUOTE GENERATION:
       Set bid/ask around mid-price estimate
       Spread = f(volatility, inventory, time-to-delivery)

    2. INVENTORY MANAGEMENT:
       Track net MW position across all products
       Adjust quotes to lean inventory back toward zero
       Hard position limits per product and aggregate

    3. MID-PRICE ESTIMATION:
       VWAP, microstructure model, or fundamental model
       Key inputs: residual demand, renewable nowcast, TSO imbalance

    4. ADVERSE SELECTION FILTER:
       Detect informed order flow (large trades, directional patterns)
       Widen spread or pause quoting when adverse selection detected

    5. TIME-TO-DELIVERY DECAY:
       As gate closure approaches, risk of holding inventory rises
       → Widen spreads, reduce size, eventually flatten position

    References:
        Avellaneda & Stoikov (2008) - Market Making HFT model
        Cartea & Jaimungal (2015) - Optimal execution in limit order books
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import deque


@dataclass
class MarketMakerParams:
    """Parameters for the intraday market maker."""
    base_spread: float          = 0.50      # base bid-ask spread [€/MWh]
    min_spread: float           = 0.10      # minimum spread [€/MWh]
    max_spread: float           = 5.00      # maximum spread [€/MWh]
    max_inventory_mw: float     = 50.0      # max net inventory [MW]
    lot_size_mw: float          = 1.0       # minimum lot [MW]
    inventory_penalty: float    = 0.05      # spread widening per MW of inventory
    vol_spread_multiplier: float= 2.0       # spread multiplier in high-vol regime
    time_decay_factor: float    = 0.5       # additional spread near gate closure
    adverse_sel_threshold: float= 3.0      # vol sigma for adverse selection detection
    market: str                 = "EPEX"
    zone: str                   = "DE_LU"


@dataclass
class Order:
    """Single limit order."""
    order_id: int
    side: str           # "bid" or "ask"
    price: float
    volume_mw: float
    product_hour: int
    timestamp: pd.Timestamp
    filled: bool = False
    fill_price: float = 0.0


class AvellanedaStoikovSpread:
    """
    Avellaneda-Stoikov optimal spread model adapted for energy ID markets.

    Optimal quotes:
        r(t) = s(t) - q * γ * σ² * (T - t)       [reservation price]
        δ_bid = 1/γ * ln(1 + γ/κ) + (q + 0.5) * γ * σ² * (T - t)
        δ_ask = 1/γ * ln(1 + γ/κ) - (q - 0.5) * γ * σ² * (T - t)

    where:
        s(t) = mid price
        q    = current inventory [MW]
        γ    = risk aversion parameter
        σ²   = price variance per unit time
        T-t  = time remaining to gate closure [hours]
        κ    = order arrival intensity

    Parameters
    ----------
    gamma   : float   Risk aversion (higher = tighter inventory management)
    sigma   : float   Price volatility [€/MWh per sqrt(hour)]
    kappa   : float   Order arrival intensity (orders per hour per unit spread)
    """

    def __init__(
        self,
        gamma: float = 0.01,
        sigma: float = 2.0,
        kappa: float = 1.5,
    ):
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa

    def reservation_price(self, mid: float, q: float, T_minus_t: float) -> float:
        """Optimal reservation price given inventory q and time remaining."""
        return mid - q * self.gamma * self.sigma**2 * T_minus_t

    def optimal_spread(self, T_minus_t: float) -> float:
        """Optimal total spread [€/MWh]."""
        gamma, sigma, kappa = self.gamma, self.sigma, self.kappa
        if gamma <= 0 or kappa <= 0:
            return 0.50
        spread = (1/gamma) * np.log(1 + gamma/kappa) + \
                 0.5 * gamma * sigma**2 * T_minus_t
        return max(0.05, spread)

    def quotes(
        self, mid: float, q: float, T_minus_t: float
    ) -> Tuple[float, float]:
        """
        Return (bid_price, ask_price) given current state.
        """
        r     = self.reservation_price(mid, q, T_minus_t)
        delta = self.optimal_spread(T_minus_t) / 2
        bid   = round(r - delta, 2)
        ask   = round(r + delta, 2)
        return bid, ask


class ContinuousIDMarketMaker:
    """
    Simulates a continuous intraday market maker.

    Simulates order flow, quote generation, fills, and inventory management
    on a tick-by-tick or minute-by-minute basis.

    Parameters
    ----------
    mid_prices      : pd.Series   Mid-price time series [€/MWh] (minute frequency).
    params          : MarketMakerParams
    as_model        : AvellanedaStoikovSpread  (optional, uses params if None)
    """

    def __init__(
        self,
        mid_prices: pd.Series,
        params: Optional[MarketMakerParams] = None,
        as_model: Optional[AvellanedaStoikovSpread] = None,
    ):
        self.mid    = mid_prices
        self.params = params or MarketMakerParams()
        self.as_mdl = as_model or AvellanedaStoikovSpread()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Spread model
    # ------------------------------------------------------------------

    def dynamic_spread(
        self,
        inventory_mw: float,
        rolling_vol: float,
        time_to_close_hours: float,
    ) -> float:
        """
        Compute dynamic bid-ask spread based on:
        1. Base spread
        2. Inventory adjustment (lean against position)
        3. Volatility regime
        4. Time-to-delivery urgency
        """
        p = self.params
        spread = p.base_spread

        # Inventory widening: bigger position → wider spread
        inv_adj = abs(inventory_mw) * p.inventory_penalty
        spread += inv_adj

        # Vol regime
        if rolling_vol > self.as_mdl.sigma * 1.5:
            spread *= p.vol_spread_multiplier

        # Time decay: widen as gate approaches
        if time_to_close_hours < 2.0:
            spread *= (1 + p.time_decay_factor * (2.0 - time_to_close_hours))

        return float(np.clip(spread, p.min_spread, p.max_spread))

    def inventory_skew(self, inventory_mw: float) -> float:
        """
        Quote skew to reduce inventory.
        Positive inventory → lower bid, lower ask (lean to sell).
        """
        max_inv = self.params.max_inventory_mw
        if max_inv <= 0:
            return 0.0
        skew = -inventory_mw / max_inv * self.params.base_spread * 0.5
        return float(np.clip(skew, -self.params.base_spread, self.params.base_spread))

    # ------------------------------------------------------------------
    # Simulate fill probability
    # ------------------------------------------------------------------

    @staticmethod
    def fill_probability(spread: float, kappa: float = 1.5) -> float:
        """
        Simplified fill probability per side per time step.
        P(fill) = exp(-kappa * half_spread)
        Based on Avellaneda-Stoikov Poisson arrival model.
        """
        return float(np.exp(-kappa * spread / 2))

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------

    def run(
        self,
        gate_closure_steps: int = 60,   # steps (minutes) before gate closure
    ) -> pd.DataFrame:
        """
        Run market making simulation.
        Each time step: generate quotes → simulate fills → update inventory.
        """
        mid     = self.mid.values
        n       = len(mid)
        params  = self.params
        rng     = np.random.default_rng(42)

        inventory   = 0.0
        cash_pnl    = 0.0
        records     = []

        roll_vol_window = 20

        for i in range(roll_vol_window, n):
            t      = self.mid.index[i]
            s      = mid[i]
            ttc    = max(0.0, (n - i) / 60)     # time to close [hours]

            # Rolling volatility
            rv = float(np.std(np.diff(mid[max(0, i-roll_vol_window):i]))) * np.sqrt(60)

            # Dynamic spread and skew
            spread = self.dynamic_spread(inventory, rv, ttc)
            skew   = self.inventory_skew(inventory)

            half  = spread / 2
            bid   = round(s - half + skew, 2)
            ask   = round(s + half + skew, 2)

            # Simulate fills (Poisson arrivals)
            p_fill = self.fill_probability(spread, kappa=self.as_mdl.kappa)

            bid_filled = (rng.random() < p_fill) and (abs(inventory) < params.max_inventory_mw)
            ask_filled = (rng.random() < p_fill) and (abs(inventory) < params.max_inventory_mw)

            # Inventory and P&L update
            if bid_filled:
                inventory += params.lot_size_mw
                cash_pnl  -= bid * params.lot_size_mw

            if ask_filled:
                inventory -= params.lot_size_mw
                cash_pnl  += ask * params.lot_size_mw

            # Mark-to-market P&L = cash + inventory * mid
            mtm_pnl = cash_pnl + inventory * s

            records.append({
                "timestamp":  t,
                "mid":        s,
                "bid":        bid,
                "ask":        ask,
                "spread":     spread,
                "inventory":  inventory,
                "cash_pnl":   cash_pnl,
                "mtm_pnl":    mtm_pnl,
                "rolling_vol":rv,
                "ttc_hours":  ttc,
                "bid_filled": bid_filled,
                "ask_filled": ask_filled,
            })

        self.results = pd.DataFrame(records).set_index("timestamp")
        return self.results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if self.results is None: raise RuntimeError("Call run() first.")
        df       = self.results
        fills    = df["bid_filled"].sum() + df["ask_filled"].sum()
        vol_mw   = fills * self.params.lot_size_mw
        final_pnl= df["mtm_pnl"].iloc[-1]
        max_inv  = df["inventory"].abs().max()
        avg_spr  = df["spread"].mean()
        return {
            "market":               self.params.market,
            "zone":                 self.params.zone,
            "total_fills":          int(fills),
            "total_volume_mw":      round(vol_mw, 1),
            "avg_spread_eur_mwh":   round(avg_spr, 4),
            "max_inventory_mw":     round(max_inv, 2),
            "final_mtm_pnl":        round(final_pnl, 2),
            "pnl_per_mw_traded":    round(final_pnl / vol_mw, 4) if vol_mw > 0 else 0.0,
            "time_flat_pct":        round((df["inventory"].abs() < 0.1).mean() * 100, 1),
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, figsize=(14, 13)) -> plt.Figure:
        if self.results is None: raise RuntimeError("Call run() first.")
        df  = self.results
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle(f"Continuous ID Market Making — {self.params.market} {self.params.zone}",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["mid"], color="#333",   lw=0.8, label="Mid")
        ax.plot(df.index, df["bid"], color="green",  lw=0.6, ls="--", label="Bid", alpha=0.7)
        ax.plot(df.index, df["ask"], color="red",    lw=0.6, ls="--", label="Ask", alpha=0.7)
        ax.fill_between(df.index, df["bid"], df["ask"], alpha=0.08, color="#9c27b0")
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("Quotes & Mid Price", fontsize=10)

        ax = axes[1]
        ax.plot(df.index, df["spread"], color="#e65100", lw=0.8)
        ax.set_ylabel("€/MWh"); ax.set_title("Dynamic Bid-Ask Spread", fontsize=10)

        ax = axes[2]
        ax.fill_between(df.index, df["inventory"], 0,
                        where=df["inventory"] > 0, color="#1565c0", alpha=0.5, label="Long inventory")
        ax.fill_between(df.index, df["inventory"], 0,
                        where=df["inventory"] < 0, color="#c62828", alpha=0.5, label="Short inventory")
        ax.axhline(self.params.max_inventory_mw,  color="red", lw=0.7, ls="--", alpha=0.7)
        ax.axhline(-self.params.max_inventory_mw, color="red", lw=0.7, ls="--", alpha=0.7)
        ax.axhline(0, color="black", lw=0.4)
        ax.legend(fontsize=8); ax.set_ylabel("MW"); ax.set_title("Inventory Position", fontsize=10)

        ax = axes[3]
        ax.fill_between(df.index, df["mtm_pnl"], 0,
                        where=df["mtm_pnl"] >= 0, color="green", alpha=0.4)
        ax.fill_between(df.index, df["mtm_pnl"], 0,
                        where=df["mtm_pnl"] < 0,  color="red",   alpha=0.4)
        ax.plot(df.index, df["mtm_pnl"], color="black", lw=0.8)
        ax.set_ylabel("€"); ax.set_title("Mark-to-Market P&L", fontsize=10)
        ax.set_xlabel("Time")
        plt.tight_layout()
        return fig


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng    = np.random.default_rng(42)
    n_min  = 480           # 8 hours of minute-by-minute data
    times  = pd.date_range("2024-01-15 08:00", periods=n_min, freq="min")

    # Simulate intraday mid-price: mean-reverting around 85 €/MWh
    mid  = [85.0]
    for _ in range(n_min - 1):
        dm = 0.05 * (85 - mid[-1]) + rng.normal(0, 0.5)
        mid.append(max(20, mid[-1] + dm))
    mid_s = pd.Series(mid, index=times, name="mid_price")

    params  = MarketMakerParams(
        base_spread=0.40, max_inventory_mw=30.0,
        lot_size_mw=1.0, inventory_penalty=0.03,
        market="EPEX", zone="DE_LU",
    )
    as_mdl  = AvellanedaStoikovSpread(gamma=0.01, sigma=2.0, kappa=1.5)
    mm      = ContinuousIDMarketMaker(mid_prices=mid_s, params=params, as_model=as_mdl)
    mm.run(gate_closure_steps=60)

    stats = mm.summary()
    print("\n=== Continuous ID Market Making — Summary ===")
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")

    # Optimal quotes example
    print("\n  Optimal quotes (AS model, q=5 MW, T-t=2h):")
    bid, ask = as_mdl.quotes(mid=85.0, q=5.0, T_minus_t=2.0)
    print(f"    Bid: {bid:.2f}  Ask: {ask:.2f}  Spread: {ask-bid:.2f}")

    fig = mm.plot()
    fig.savefig("continuous_id_mm.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → continuous_id_mm.png")
