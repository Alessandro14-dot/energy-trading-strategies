"""
Gas Hub Spread Arbitrage — TTF vs NCG / TTF vs NBP
====================================================
Exploits price differentials between European natural gas trading hubs.

Hub overview:
    TTF  (Title Transfer Facility) — Netherlands, most liquid EU gas hub
    NCG  (NetConnect Germany)      — Germany, second most liquid
    NBP  (National Balancing Point)— UK, historically first major hub
    PSV  (Punto di Scambio Virtuale)— Italy, premium hub (pipeline bottleneck)
    PEG  (Point d'Échange de Gaz)  — France

Market rationale:
    In theory, arbitrage should keep hub prices aligned (minus transport cost).
    In practice, spreads persist due to:
        - Pipeline capacity constraints
        - Storage and injection/withdrawal dynamics
        - LNG import competition (NBP more LNG-exposed)
        - Regulatory and balancing differences
        - Liquidity differences (TTF >> others)
        - Seasonal demand patterns by country

Key spread pairs:
    TTF–NCG   : typically small (pipeline well-connected), ~0.1–0.5 €/MWh
    TTF–NBP   : larger, driven by LNG flows, Brexit effects, UK storage
    TTF–PSV   : Italy premium reflects southern pipeline constraints

Trading:
    - Statistical mean-reversion on hub spreads
    - Fundamental overlay: storage differentials, LNG send-out
    - Regime detection: contango vs backwardation by hub

Units:
    TTF / NCG / PSV / PEG : €/MWh
    NBP                   : p/therm (must convert: 1 therm ≈ 29.307 kWh)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class HubSpreadConfig:
    """Configuration for gas hub spread strategy."""
    hub_a: str              = "TTF"
    hub_b: str              = "NCG"
    lookback: int           = 20
    entry_zscore: float     = 1.8
    exit_zscore: float      = 0.4
    stop_zscore: float      = 3.5
    transport_cost: float   = 0.15      # €/MWh one-way pipeline tariff


def convert_nbp_to_eur_mwh(nbp_pence_therm: pd.Series, gbp_eur: pd.Series) -> pd.Series:
    """
    Convert NBP prices from pence/therm to €/MWh.
    1 therm = 29.307 kWh = 0.029307 MWh
    price (€/MWh) = price(p/therm) / 100 * fx(£/€) / 0.029307
    """
    gbp_per_therm = nbp_pence_therm / 100
    eur_per_therm = gbp_per_therm * gbp_eur
    eur_per_mwh   = eur_per_therm / 0.029307
    return eur_per_mwh.rename("NBP_eur_mwh")


class GasHubSpread:
    """
    Gas hub spread arbitrage strategy.

    Parameters
    ----------
    hub_a_prices      : pd.Series   Prices at hub A (reference, typically TTF) [€/MWh].
    hub_b_prices      : pd.Series   Prices at hub B [€/MWh].
    storage_delta     : pd.Series   Optional storage level difference A-B [%].
    lng_sendout       : pd.Series   Optional LNG send-out at hub B [GWh/day].
    config            : HubSpreadConfig
    """

    def __init__(
        self,
        hub_a_prices: pd.Series,
        hub_b_prices: pd.Series,
        storage_delta: Optional[pd.Series] = None,
        lng_sendout: Optional[pd.Series] = None,
        config: Optional[HubSpreadConfig] = None,
    ):
        self.hub_a        = hub_a_prices.rename(config.hub_a if config else "hub_A")
        self.hub_b        = hub_b_prices.rename(config.hub_b if config else "hub_B")
        self.stor_delta   = storage_delta
        self.lng_sendout  = lng_sendout
        self.cfg          = config or HubSpreadConfig()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Spread
    # ------------------------------------------------------------------

    def compute_spread(self) -> pd.Series:
        """Hub_A - Hub_B [€/MWh]."""
        s = self.hub_a - self.hub_b
        s.name = "hub_spread"
        return s

    def fair_value_spread(self) -> pd.Series:
        """
        Fair value spread = transport cost (one-way).
        If hub_A is the source and hub_B is the destination,
        the fair spread is ~transport_cost. Deviations = opportunity.
        """
        return pd.Series(self.cfg.transport_cost,
                         index=self.hub_a.index, name="fair_value")

    def net_opportunity(self, spread: pd.Series) -> pd.Series:
        """Spread net of transport cost and round-trip friction."""
        rt = self.cfg.transport_cost * 2   # round trip
        net = spread.abs() - rt
        net.name = "net_opportunity"
        return net.clip(lower=0)

    # ------------------------------------------------------------------
    # Fundamental overlays
    # ------------------------------------------------------------------

    def storage_signal(self) -> Optional[pd.Series]:
        """
        Directional bias from storage differential.
        Positive delta (hub_A more full) → bearish hub_A → lean short spread.
        """
        if self.stor_delta is None:
            return None
        sig = pd.Series(0, index=self.stor_delta.index, name="stor_signal")
        sig[self.stor_delta >  15] = -1   # hub_A relatively full → short A vs B
        sig[self.stor_delta < -15] =  1   # hub_A relatively empty → long A vs B
        return sig

    def lng_signal(self) -> Optional[pd.Series]:
        """
        High LNG send-out at hub_B → bearish hub_B → bullish spread (A-B widens).
        """
        if self.lng_sendout is None:
            return None
        threshold = self.lng_sendout.rolling(30).mean() + self.lng_sendout.rolling(30).std()
        sig = pd.Series(0, index=self.lng_sendout.index, name="lng_signal")
        sig[self.lng_sendout > threshold] = 1   # excess LNG at B → B cheaper → spread widens
        return sig

    # ------------------------------------------------------------------
    # Run strategy
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Generate z-score signals with optional fundamental overlays."""
        spread    = self.compute_spread()
        fv        = self.fair_value_spread()
        net_opp   = self.net_opportunity(spread)
        roll_mean = spread.rolling(self.cfg.lookback).mean()
        roll_std  = spread.rolling(self.cfg.lookback).std()
        zscore    = (spread - roll_mean) / roll_std.replace(0, np.nan)
        stor_sig  = self.storage_signal()
        lng_sig   = self.lng_signal()

        position = pd.Series(0.0, index=spread.index)
        current  = 0

        for i in range(self.cfg.lookback, len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z):
                continue

            # Combine fundamental signals (simple sum, capped at ±1)
            bias = 0
            if stor_sig is not None:
                bias += stor_sig.iloc[i]
            if lng_sig is not None:
                bias += lng_sig.iloc[i]
            bias = np.clip(bias, -1, 1)

            if current == 0:
                if z > self.cfg.entry_zscore and bias <= 0:
                    current = -1    # spread too wide → short A, long B
                elif z < -self.cfg.entry_zscore and bias >= 0:
                    current = 1     # spread too narrow → long A, short B
            elif current == 1:
                if z >= -self.cfg.exit_zscore or z <= -self.cfg.stop_zscore:
                    current = 0
            elif current == -1:
                if z <= self.cfg.exit_zscore or z >= self.cfg.stop_zscore:
                    current = 0

            position.iloc[i] = current

        daily_pnl = position.shift(1) * spread.diff()

        self.results = pd.DataFrame({
            "hub_A":        self.hub_a,
            "hub_B":        self.hub_b,
            "hub_spread":   spread,
            "fair_value":   fv,
            "net_opp":      net_opp,
            "stor_delta":   self.stor_delta if self.stor_delta is not None else np.nan,
            "lng_sendout":  self.lng_sendout if self.lng_sendout is not None else np.nan,
            "roll_mean":    roll_mean,
            "zscore":       zscore,
            "position":     position,
            "daily_pnl":    daily_pnl,
            "cum_pnl":      daily_pnl.cumsum(),
        })
        return self.results

    # ------------------------------------------------------------------
    # Spread decomposition
    # ------------------------------------------------------------------

    def spread_decomposition(self) -> pd.DataFrame:
        """
        Decompose the spread into:
        - Structural component (rolling long-term mean)
        - Cyclical component (seasonal)
        - Residual (noise / trading signal)
        """
        spread = self.compute_spread()
        structural = spread.rolling(120, min_periods=30).mean()
        doy        = spread.index.dayofyear
        seasonal_m = spread.groupby(doy).transform("mean")
        residual   = spread - structural - (seasonal_m - seasonal_m.mean())

        return pd.DataFrame({
            "spread":      spread,
            "structural":  structural,
            "seasonal":    seasonal_m - seasonal_m.mean(),
            "residual":    residual,
        }).round(3)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        pnl    = self.results["daily_pnl"].dropna()
        spread = self.results["hub_spread"].dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        dd     = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        return {
            "spread_pair":          f"{self.cfg.hub_a} - {self.cfg.hub_b}",
            "spread_mean":          round(spread.mean(), 3),
            "spread_std":           round(spread.std(), 3),
            "spread_abs_mean":      round(spread.abs().mean(), 3),
            "transport_cost":       self.cfg.transport_cost,
            "net_opp_days_pct":     round((self.results["net_opp"] > 0).mean() * 100, 1),
            "total_pnl":            round(pnl.sum(), 2),
            "sharpe_ratio":         round(sharpe, 3),
            "max_drawdown":         round(dd, 2),
            "win_rate":             round((pnl > 0).mean(), 3),
            "n_long":               int((self.results["position"] == 1).sum()),
            "n_short":              int((self.results["position"] == -1).sum()),
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, figsize=(14, 11)) -> plt.Figure:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        df  = self.results
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle(f"Gas Hub Spread — {self.cfg.hub_a} vs {self.cfg.hub_b}",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["hub_A"], label=self.cfg.hub_a, color="#1565c0", lw=1.0)
        ax.plot(df.index, df["hub_B"], label=self.cfg.hub_b, color="#2e7d32", lw=1.0)
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("Hub Prices", fontsize=10)

        ax = axes[1]
        ax.plot(df.index, df["hub_spread"], color="#333", lw=0.9, label="Spread")
        ax.plot(df.index, df["roll_mean"],  color="#9c27b0", lw=0.8, ls="--", label="Rolling Mean")
        ax.axhline(self.cfg.transport_cost,  color="orange", lw=0.7, ls=":", label="±Transport cost")
        ax.axhline(-self.cfg.transport_cost, color="orange", lw=0.7, ls=":")
        ax.axhline(0, color="black", lw=0.4)
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("Hub Spread", fontsize=10)

        ax = axes[2]
        ax.plot(df.index, df["zscore"], color="#555", lw=0.8)
        ax.axhline(self.cfg.entry_zscore,  color="red",   lw=0.8, ls="--")
        ax.axhline(-self.cfg.entry_zscore, color="green", lw=0.8, ls="--")
        ax.axhline(0, color="black", lw=0.3)
        longs  = df[df["position"] == 1]
        shorts = df[df["position"] == -1]
        ax.scatter(longs.index,  longs["zscore"],  s=14, color="green", zorder=5, label="Long")
        ax.scatter(shorts.index, shorts["zscore"], s=14, color="red",   zorder=5, label="Short")
        ax.legend(fontsize=8); ax.set_ylabel("Z-score"); ax.set_title("Signal", fontsize=10)

        ax = axes[3]
        ax.fill_between(df.index, df["cum_pnl"], 0,
                        where=df["cum_pnl"] >= 0, color="green", alpha=0.4)
        ax.fill_between(df.index, df["cum_pnl"], 0,
                        where=df["cum_pnl"] < 0,  color="red",   alpha=0.4)
        ax.plot(df.index, df["cum_pnl"], color="black", lw=0.8)
        ax.set_ylabel("€/MWh cumul."); ax.set_title("Cumulative P&L", fontsize=10)
        ax.set_xlabel("Date")

        plt.tight_layout()
        return fig


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng   = np.random.default_rng(21)
    n     = 700
    dates = pd.date_range("2021-01-01", periods=n, freq="B")

    common  = np.cumsum(rng.normal(0, 1.5, n))
    ttf     = np.clip(50 + common + rng.normal(0, 0.5, n), 15, 300)
    ncg     = np.clip(50 + common + rng.normal(0, 0.6, n) + 0.3, 15, 300)
    stor_d  = rng.normal(0, 8, n)          # storage differential
    lng     = np.clip(50 + rng.normal(0, 20, n), 0, 200)  # LNG send-out GWh/d

    cfg = HubSpreadConfig(hub_a="TTF", hub_b="NCG", lookback=20,
                          entry_zscore=1.8, transport_cost=0.15)

    strat = GasHubSpread(
        hub_a_prices  = pd.Series(ttf,    index=dates),
        hub_b_prices  = pd.Series(ncg,    index=dates),
        storage_delta = pd.Series(stor_d, index=dates),
        lng_sendout   = pd.Series(lng,    index=dates),
        config=cfg,
    )

    results = strat.run()
    stats   = strat.summary()
    print("\n=== Gas Hub Spread (TTF-NCG) — Summary ===")
    for k, v in stats.items():
        print(f"  {k:28s}: {v}")

    fig = strat.plot()
    fig.savefig("ttf_ncg_spread.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → ttf_ncg_spread.png")
