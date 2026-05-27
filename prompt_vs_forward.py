"""
Prompt vs Forward Calendar Spread
===================================
Exploits the time-premium difference between the prompt contract
(Day-Ahead or Week-Ahead) and the front-month forward contract.

Market rationale:
    In energy markets the forward curve is rarely flat. The spread between
    the prompt price and the M+1 (or Q+1) forward reflects:
        - Risk premium:       producers/consumers hedge forward → pay premium
        - Storage / carry:    cost of holding gas or capacity over time
        - Demand uncertainty: weather, industrial load surprises
        - Liquidity premium:  prompt market is more liquid → mean-reverts faster

Strategy logic:
    1. Compute daily spread:  S(t) = Prompt(t) - Forward(t)
    2. Model S as mean-reverting around a rolling or seasonal mean
    3. Enter when S deviates > n*sigma from mean (z-score signal)
    4. Exit when S returns toward mean

Typical pairs:
    - DA Power vs M+1 Power  (EPEX SPOT vs EEX/ICE forward)
    - DA Gas  vs M+1 Gas     (spot TTF vs TTF M+1)
    - WD Gas  vs Q+1 Gas     (within-day vs quarterly)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class PromptForwardConfig:
    """Configuration for the prompt vs forward spread strategy."""
    lookback: int = 20                          # rolling window for z-score
    entry_zscore: float = 1.8                   # entry threshold
    exit_zscore: float = 0.3                    # exit threshold
    stop_zscore: float = 4.0                    # stop-loss threshold
    seasonality_window: int = 30                # window for seasonal adjustment
    use_seasonal_mean: bool = False             # use seasonal vs simple rolling mean
    contract_pair: str = "DA vs M+1 Power"      # label for plots/reports


class PromptForwardSpread:
    """
    Prompt vs Forward calendar spread strategy.

    Parameters
    ----------
    prompt  : pd.Series   Prompt prices (Day-Ahead or spot) [€/MWh].
    forward : pd.Series   Forward prices (M+1 or Q+1) [€/MWh].
    config  : PromptForwardConfig
    """

    def __init__(
        self,
        prompt: pd.Series,
        forward: pd.Series,
        config: Optional[PromptForwardConfig] = None,
    ):
        self.prompt = prompt.rename("prompt")
        self.forward = forward.rename("forward")
        self.cfg = config or PromptForwardConfig()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Spread & statistics
    # ------------------------------------------------------------------

    def compute_spread(self) -> pd.Series:
        """Raw spread: Prompt - Forward [€/MWh]."""
        s = self.prompt - self.forward
        s.name = "spread"
        return s

    def compute_basis(self) -> pd.Series:
        """
        Normalised basis: spread / forward  [dimensionless].
        Useful for comparing across different price regimes.
        """
        b = (self.prompt - self.forward) / self.forward.replace(0, np.nan)
        b.name = "basis"
        return b

    def rolling_stats(self, spread: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Return (rolling_mean, rolling_std, z_score) for the spread."""
        w = self.cfg.lookback

        if self.cfg.use_seasonal_mean:
            # Day-of-year seasonal mean as baseline
            doy = spread.index.dayofyear
            seasonal_mean = spread.groupby(doy).transform("mean")
            roll_mean = seasonal_mean.rolling(self.cfg.seasonality_window, center=True,
                                              min_periods=5).mean()
        else:
            roll_mean = spread.rolling(w).mean()

        roll_std = spread.rolling(w).std()
        zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)
        return roll_mean, roll_std, zscore

    # ------------------------------------------------------------------
    # Signal generation & backtest
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Generate signals and compute daily P&L.

        Position interpretation:
            +1 = long spread  (long prompt, short forward) — expect spread to widen
            -1 = short spread (short prompt, long forward) — expect spread to narrow
        """
        spread = self.compute_spread()
        basis  = self.compute_basis()
        roll_mean, roll_std, zscore = self.rolling_stats(spread)

        position = pd.Series(0.0, index=spread.index)
        current  = 0

        for i in range(self.cfg.lookback, len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z):
                continue
            if current == 0:
                if z > self.cfg.entry_zscore:
                    current = -1    # spread too wide → short, expect reversion
                elif z < -self.cfg.entry_zscore:
                    current = 1     # spread too narrow → long, expect widening
            elif current == 1:
                if z >= -self.cfg.exit_zscore or z <= -self.cfg.stop_zscore:
                    current = 0
            elif current == -1:
                if z <= self.cfg.exit_zscore or z >= self.cfg.stop_zscore:
                    current = 0
            position.iloc[i] = current

        daily_pnl = position.shift(1) * spread.diff()
        cum_pnl   = daily_pnl.cumsum()

        self.results = pd.DataFrame({
            "prompt":      self.prompt,
            "forward":     self.forward,
            "spread":      spread,
            "basis":       basis,
            "roll_mean":   roll_mean,
            "roll_std":    roll_std,
            "zscore":      zscore,
            "position":    position,
            "daily_pnl":   daily_pnl,
            "cum_pnl":     cum_pnl,
        })
        return self.results

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        pnl = self.results["daily_pnl"].dropna()
        spread = self.results["spread"].dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        dd = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        return {
            "contract_pair":    self.cfg.contract_pair,
            "spread_mean":      round(spread.mean(), 2),
            "spread_std":       round(spread.std(), 2),
            "spread_min":       round(spread.min(), 2),
            "spread_max":       round(spread.max(), 2),
            "contango_pct":     round((spread < 0).mean() * 100, 1),   # % days forward > prompt
            "backwardation_pct":round((spread > 0).mean() * 100, 1),   # % days prompt > forward
            "total_pnl":        round(pnl.sum(), 2),
            "sharpe_ratio":     round(sharpe, 3),
            "max_drawdown":     round(dd, 2),
            "win_rate":         round((pnl > 0).mean(), 3),
            "n_long":           int((self.results["position"] == 1).sum()),
            "n_short":          int((self.results["position"] == -1).sum()),
        }

    def forward_curve_shape(self) -> str:
        """Return dominant curve shape based on average spread."""
        if self.results is None:
            self.run()
        avg = self.results["spread"].mean()
        if avg > 1.0:
            return "Backwardation (prompt > forward) — typical in supply-tightness regimes"
        elif avg < -1.0:
            return "Contango (forward > prompt) — typical in storage-build / oversupply regimes"
        else:
            return "Flat curve — balanced supply/demand"

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, figsize=(14, 11)) -> plt.Figure:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        df = self.results
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle(f"Prompt vs Forward — {self.cfg.contract_pair}", fontsize=13, fontweight="bold")

        # Panel 1: Prices
        ax = axes[0]
        ax.plot(df.index, df["prompt"],  label="Prompt",  color="#1a73e8", lw=1.0)
        ax.plot(df.index, df["forward"], label="Forward", color="#e67700", lw=1.0, ls="--")
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("Prices", fontsize=10)

        # Panel 2: Spread
        ax = axes[1]
        ax.plot(df.index, df["spread"],    label="Spread",        color="#333", lw=0.9)
        ax.plot(df.index, df["roll_mean"], label="Rolling Mean",  color="#9c27b0", lw=0.8, ls="--")
        ax.fill_between(df.index,
                        df["roll_mean"] + df["roll_std"],
                        df["roll_mean"] - df["roll_std"],
                        alpha=0.12, color="#9c27b0", label="±1σ")
        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("Calendar Spread", fontsize=10)

        # Panel 3: Z-score & signals
        ax = axes[2]
        ax.plot(df.index, df["zscore"], color="#555", lw=0.8)
        for lvl, col in [(self.cfg.entry_zscore, "red"), (-self.cfg.entry_zscore, "green")]:
            ax.axhline(lvl, color=col, lw=0.8, ls="--", alpha=0.8)
        ax.axhline(0, color="black", lw=0.3)
        longs  = df[df["position"] == 1]
        shorts = df[df["position"] == -1]
        ax.scatter(longs.index,  longs["zscore"],  s=14, color="green", zorder=5, label="Long")
        ax.scatter(shorts.index, shorts["zscore"], s=14, color="red",   zorder=5, label="Short")
        ax.legend(fontsize=8); ax.set_ylabel("Z-score"); ax.set_title("Signal", fontsize=10)

        # Panel 4: P&L
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
    rng   = np.random.default_rng(42)
    n     = 700
    dates = pd.date_range("2021-01-01", periods=n, freq="B")

    # Seasonal forward curve: slight contango base, with winter premium
    doy         = np.array([d.dayofyear for d in dates])
    seasonal    = 3.0 * np.cos(2 * np.pi * (doy - 355) / 365)   # winter peak
    forward_px  = 80 + np.cumsum(rng.normal(0, 1.2, n)) + seasonal
    forward_px  = np.clip(forward_px, 30, 400)
    prompt_px   = forward_px + seasonal * 0.5 + rng.normal(0, 3, n)
    prompt_px   = np.clip(prompt_px, 20, 500)

    cfg = PromptForwardConfig(
        lookback=25,
        entry_zscore=1.8,
        exit_zscore=0.4,
        contract_pair="DA Power vs M+1 Power (DE)",
    )
    strat = PromptForwardSpread(
        prompt=pd.Series(prompt_px, index=dates),
        forward=pd.Series(forward_px, index=dates),
        config=cfg,
    )

    results = strat.run()
    stats   = strat.summary()

    print("\n=== Prompt vs Forward — Summary ===")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")
    print(f"\n  Curve shape: {strat.forward_curve_shape()}")

    fig = strat.plot()
    fig.savefig("prompt_vs_forward.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → prompt_vs_forward.png")
