"""
Winter / Summer Seasonal Strip Spread
=======================================
Exploits the structural price difference between winter and summer
energy contracts — one of the most liquid and actively traded
calendar spreads in European power and gas markets.

Market rationale:
    Winter power demand (heating, lighting) is structurally higher than summer.
    Winter gas demand for heating creates a seasonal premium.
    The W/S spread reflects:
        - Heating demand premium (gas, power)
        - Hydro availability (summer hydro → summer power cheaper)
        - Storage injection/withdrawal cycle
        - Renewable seasonality (wind stronger winter, solar stronger summer)

Contract definitions:
    Winter strip : Oct–Mar delivery (Q4 + Q1)
    Summer strip : Apr–Sep delivery (Q2 + Q3)
    W/S Spread   : Winter Price - Summer Price

    In gas:   typically quoted as Win/Sum spread on TTF
    In power: Winter Cal vs Summer Cal, or Q4+Q1 vs Q2+Q3

Trading signals:
    - Compare current W/S spread to historical distribution
    - Enter long W/S when spread is unusually low (buy winter, sell summer)
    - Enter short W/S when spread is unusually high (sell winter, buy summer)
    - Fundamental overlay: storage levels, temperature forecasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class WSSpreadConfig:
    """Configuration for winter/summer spread strategy."""
    lookback: int = 40                    # rolling window for z-score
    entry_zscore: float = 1.6
    exit_zscore: float = 0.4
    stop_zscore: float = 3.5
    market: Literal["power", "gas"] = "power"
    hub: str = "DE"


class WinterSummerSpread:
    """
    Winter vs Summer strip spread strategy.

    Parameters
    ----------
    winter_prices : pd.Series   Winter strip prices [€/MWh].
    summer_prices : pd.Series   Summer strip prices [€/MWh].
    storage_levels: pd.Series   Optional storage fill % (0–100) for fundamental overlay.
    config        : WSSpreadConfig
    """

    def __init__(
        self,
        winter_prices: pd.Series,
        summer_prices: pd.Series,
        storage_levels: Optional[pd.Series] = None,
        config: Optional[WSSpreadConfig] = None,
    ):
        self.winter  = winter_prices.rename("winter")
        self.summer  = summer_prices.rename("summer")
        self.storage = storage_levels
        self.cfg     = config or WSSpreadConfig()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Spread
    # ------------------------------------------------------------------

    def compute_ws_spread(self) -> pd.Series:
        """Winter - Summer spread [€/MWh]."""
        s = self.winter - self.summer
        s.name = "ws_spread"
        return s

    def seasonal_adjustment(self, spread: pd.Series) -> pd.Series:
        """
        Remove intra-year seasonality from the spread series.
        Returns the residual (deseasonalised) spread.
        """
        month = spread.index.month
        monthly_mean = spread.groupby(month).transform("mean")
        return (spread - monthly_mean).rename("ws_spread_adj")

    # ------------------------------------------------------------------
    # Fundamental overlay: storage signal
    # ------------------------------------------------------------------

    def storage_signal(self) -> Optional[pd.Series]:
        """
        Convert storage fill level to a directional bias.

        Logic (gas market):
            High storage (>80%) → bearish winter premium → lean short W/S spread
            Low  storage (<40%) → bullish winter premium → lean long  W/S spread

        Returns a series in {-1, 0, +1}.
        """
        if self.storage is None:
            return None
        sig = pd.Series(0, index=self.storage.index, name="storage_signal")
        sig[self.storage > 80] = -1
        sig[self.storage < 40] =  1
        return sig

    # ------------------------------------------------------------------
    # Run strategy
    # ------------------------------------------------------------------

    def run(self, use_storage_overlay: bool = True) -> pd.DataFrame:
        """
        Generate signals combining z-score mean-reversion
        with optional storage fundamental overlay.
        """
        spread      = self.compute_ws_spread()
        spread_adj  = self.seasonal_adjustment(spread)
        roll_mean   = spread.rolling(self.cfg.lookback).mean()
        roll_std    = spread.rolling(self.cfg.lookback).std()
        zscore      = (spread - roll_mean) / roll_std.replace(0, np.nan)
        stor_signal = self.storage_signal() if use_storage_overlay else None

        position = pd.Series(0.0, index=spread.index)
        current  = 0

        for i in range(self.cfg.lookback, len(zscore)):
            z   = zscore.iloc[i]
            if np.isnan(z):
                continue

            # Storage overlay: only take trades aligned with fundamental bias
            bias = 0
            if stor_signal is not None:
                bias = stor_signal.iloc[i]

            if current == 0:
                if z < -self.cfg.entry_zscore and (bias >= 0):
                    current = 1    # long W/S: spread too narrow, expect widening
                elif z > self.cfg.entry_zscore and (bias <= 0):
                    current = -1   # short W/S: spread too wide, expect narrowing
            elif current == 1:
                if z >= -self.cfg.exit_zscore or z <= -self.cfg.stop_zscore:
                    current = 0
            elif current == -1:
                if z <= self.cfg.exit_zscore or z >= self.cfg.stop_zscore:
                    current = 0
            position.iloc[i] = current

        daily_pnl = position.shift(1) * spread.diff()

        self.results = pd.DataFrame({
            "winter":       self.winter,
            "summer":       self.summer,
            "ws_spread":    spread,
            "ws_spread_adj":spread_adj,
            "storage":      self.storage if self.storage is not None else np.nan,
            "roll_mean":    roll_mean,
            "zscore":       zscore,
            "stor_signal":  stor_signal if stor_signal is not None else np.nan,
            "position":     position,
            "daily_pnl":    daily_pnl,
            "cum_pnl":      daily_pnl.cumsum(),
        })
        return self.results

    # ------------------------------------------------------------------
    # Historical seasonality analysis
    # ------------------------------------------------------------------

    def monthly_spread_profile(self) -> pd.DataFrame:
        """
        Average W/S spread by calendar month.
        Useful for understanding when the spread is typically
        at its seasonal peak or trough.
        """
        spread = self.compute_ws_spread()
        df = pd.DataFrame({"spread": spread, "month": spread.index.month})
        profile = df.groupby("month")["spread"].agg(["mean", "std", "min", "max"])
        profile.index = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
        return profile.round(2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        pnl    = self.results["daily_pnl"].dropna()
        spread = self.results["ws_spread"].dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        dd     = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        return {
            "market":           f"{self.cfg.market} — {self.cfg.hub}",
            "ws_spread_mean":   round(spread.mean(), 2),
            "ws_spread_std":    round(spread.std(), 2),
            "ws_spread_max":    round(spread.max(), 2),
            "ws_spread_min":    round(spread.min(), 2),
            "total_pnl":        round(pnl.sum(), 2),
            "sharpe_ratio":     round(sharpe, 3),
            "max_drawdown":     round(dd, 2),
            "win_rate":         round((pnl > 0).mean(), 3),
            "n_long":           int((self.results["position"] == 1).sum()),
            "n_short":          int((self.results["position"] == -1).sum()),
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, figsize=(14, 12)) -> plt.Figure:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        df  = self.results
        has_stor = not df["storage"].isna().all()
        nrows = 5 if has_stor else 4

        fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True)
        fig.suptitle(f"Winter/Summer Spread — {self.cfg.market.upper()} {self.cfg.hub}",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["winter"], label="Winter Strip", color="#1565c0", lw=1.0)
        ax.plot(df.index, df["summer"], label="Summer Strip", color="#f9a825", lw=1.0)
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("Strip Prices", fontsize=10)

        ax = axes[1]
        ax.plot(df.index, df["ws_spread"], color="#333", lw=0.9, label="W/S Spread")
        ax.plot(df.index, df["roll_mean"], color="#9c27b0", lw=0.8, ls="--", label="Rolling Mean")
        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.fill_between(df.index, df["ws_spread"], 0,
                        where=df["ws_spread"] > 0, color="#1565c0", alpha=0.15, label="Backwardation")
        ax.fill_between(df.index, df["ws_spread"], 0,
                        where=df["ws_spread"] < 0, color="#f9a825", alpha=0.15, label="Contango")
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("W/S Spread", fontsize=10)

        row = 2
        if has_stor:
            ax = axes[row]
            ax.plot(df.index, df["storage"], color="#2e7d32", lw=0.9)
            ax.axhline(80, color="red",   lw=0.7, ls="--", alpha=0.7, label="80% (bearish W)")
            ax.axhline(40, color="green", lw=0.7, ls="--", alpha=0.7, label="40% (bullish W)")
            ax.set_ylabel("%"); ax.set_title("Storage Fill Level", fontsize=10)
            ax.legend(fontsize=8); row += 1

        ax = axes[row]
        ax.plot(df.index, df["zscore"], color="#555", lw=0.8)
        ax.axhline(self.cfg.entry_zscore,  color="red",   lw=0.8, ls="--")
        ax.axhline(-self.cfg.entry_zscore, color="green", lw=0.8, ls="--")
        ax.axhline(0, color="black", lw=0.3)
        longs  = df[df["position"] == 1]
        shorts = df[df["position"] == -1]
        ax.scatter(longs.index,  longs["zscore"],  s=14, color="green", zorder=5)
        ax.scatter(shorts.index, shorts["zscore"], s=14, color="red",   zorder=5)
        ax.set_ylabel("Z-score"); ax.set_title("Signal", fontsize=10); row += 1

        ax = axes[row]
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
    rng   = np.random.default_rng(12)
    n     = 800
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Winter strip carries seasonal premium + random walk
    base    = 70 + np.cumsum(rng.normal(0, 1.0, n))
    doy     = np.array([d.dayofyear for d in dates])
    seas    = 12 * np.cos(2 * np.pi * (doy - 15) / 365)    # peak in Jan
    winter  = np.clip(base + seas + rng.normal(0, 2, n), 30, 400)
    summer  = np.clip(base - seas * 0.5 + rng.normal(0, 2, n), 20, 300)

    # Synthetic storage levels (seasonal fill/drain)
    stor    = 50 + 35 * np.sin(2 * np.pi * (doy - 90) / 365) + rng.normal(0, 3, n)
    stor    = np.clip(stor, 5, 100)

    cfg = WSSpreadConfig(lookback=40, entry_zscore=1.6, market="power", hub="DE")
    strat = WinterSummerSpread(
        winter_prices  = pd.Series(winter, index=dates),
        summer_prices  = pd.Series(summer, index=dates),
        storage_levels = pd.Series(stor,   index=dates),
        config=cfg,
    )

    results = strat.run(use_storage_overlay=True)
    stats   = strat.summary()

    print("\n=== Winter/Summer Spread — Summary ===")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    print("\n  Monthly Spread Profile:")
    print(strat.monthly_spread_profile().to_string())

    fig = strat.plot()
    fig.savefig("winter_summer_spread.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → winter_summer_spread.png")
