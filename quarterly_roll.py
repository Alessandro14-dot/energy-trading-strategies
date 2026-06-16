"""
Quarterly Roll Strategy
========================
Exploits the price dynamics around the quarterly contract roll in
European power and gas forward markets.

Market rationale:
    Energy forward curves are structured in quarterly contracts (Q1–Q4).
    As the front quarter approaches expiry, several effects occur:

    1. Roll premium / discount:
       The front quarter trades at a premium or discount to Q+1
       as liquidity migrates to the new front contract.

    2. Calendar roll arbitrage:
       The spread between expiring Q and next Q can deviate from
       fair value due to hedging flows, liquidity imbalances,
       and position unwinding by market participants.

    3. Shape trading:
       Buying the back quarter and selling the front (or vice versa)
       based on the forward curve shape (contango vs backwardation).

Contract structure (European standard):
    Q1 = Jan–Mar    Q2 = Apr–Jun
    Q3 = Jul–Sep    Q4 = Oct–Dec

    Roll dates: typically last trading day of the preceding quarter
    Most liquid rolls: Q4→Q1 (winter roll), Q1→Q2 (spring roll)

Strategy variants implemented:
    A) Roll Spread Mean-Reversion: z-score on Q_front vs Q_back spread
    B) Roll Timing:                enter/exit around known roll dates
    C) Curve Shape:                contango vs backwardation regime filter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Literal


# ─── Quarter helpers ──────────────────────────────────────────────────────────

QUARTER_MONTHS = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}

def quarter_label(date: pd.Timestamp) -> str:
    return f"Q{date.quarter} {date.year}"

def days_to_roll(date: pd.Timestamp) -> int:
    """Business days remaining until end of current quarter."""
    q_end_month = QUARTER_MONTHS[date.quarter][1]
    q_end = pd.Timestamp(year=date.year, month=q_end_month,
                         day=1) + pd.offsets.MonthEnd(0)
    bdays = pd.bdate_range(date, q_end)
    return max(len(bdays) - 1, 0)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class QuarterlyRollConfig:
    """Parameters for the quarterly roll strategy."""
    lookback: int = 30                          # rolling window for z-score
    entry_zscore: float = 1.7
    exit_zscore: float = 0.4
    stop_zscore: float = 3.5
    roll_window_days: int = 10                  # days around roll to apply timing overlay
    use_roll_timing: bool = True                # activate roll timing overlay
    use_curve_shape: bool = True                # activate curve shape regime filter
    market: str = "Power DE"


# ─── Main class ───────────────────────────────────────────────────────────────

class QuarterlyRollStrategy:
    """
    Quarterly roll spread strategy.

    Parameters
    ----------
    front_quarter  : pd.Series   Front quarter (Q_n) prices [€/MWh].
    back_quarter   : pd.Series   Back quarter (Q_n+1) prices [€/MWh].
    cal_year       : pd.Series   Optional Cal year price for curve shape context.
    config         : QuarterlyRollConfig
    """

    def __init__(
        self,
        front_quarter: pd.Series,
        back_quarter: pd.Series,
        cal_year: Optional[pd.Series] = None,
        config: Optional[QuarterlyRollConfig] = None,
    ):
        self.front   = front_quarter.rename("front_Q")
        self.back    = back_quarter.rename("back_Q")
        self.cal     = cal_year
        self.cfg     = config or QuarterlyRollConfig()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Spread & analytics
    # ------------------------------------------------------------------

    def compute_spread(self) -> pd.Series:
        """Roll spread: Front Q - Back Q [€/MWh]."""
        s = self.front - self.back
        s.name = "roll_spread"
        return s

    def curve_shape(self, spread: pd.Series) -> pd.Series:
        """
        Classify each day as contango (-1), flat (0), or backwardation (+1)
        based on rolling average of the roll spread.
        """
        roll_avg = spread.rolling(self.cfg.lookback).mean()
        shape = pd.Series(0, index=spread.index, name="curve_shape")
        shape[roll_avg > 1.0]  =  1    # backwardation: front > back
        shape[roll_avg < -1.0] = -1    # contango:      front < back
        return shape

    def roll_proximity(self) -> pd.Series:
        """
        Returns days-to-roll for each date.
        Values close to 0 indicate imminent roll.
        """
        return pd.Series(
            [days_to_roll(d) for d in self.front.index],
            index=self.front.index,
            name="days_to_roll",
        )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Generate trading signals combining:
        - Z-score mean reversion on the roll spread
        - Roll timing overlay (avoid entering near roll date)
        - Curve shape regime filter
        """
        spread      = self.compute_spread()
        roll_mean   = spread.rolling(self.cfg.lookback).mean()
        roll_std    = spread.rolling(self.cfg.lookback).std()
        zscore      = (spread - roll_mean) / roll_std.replace(0, np.nan)
        shape       = self.curve_shape(spread)
        dtr         = self.roll_proximity()

        position = pd.Series(0.0, index=spread.index)
        current  = 0

        for i in range(self.cfg.lookback, len(zscore)):
            z   = zscore.iloc[i]
            d   = dtr.iloc[i]
            sh  = shape.iloc[i]
            if np.isnan(z):
                continue

            # Roll timing: avoid new entries within roll_window_days of expiry
            near_roll = self.cfg.use_roll_timing and (d <= self.cfg.roll_window_days)

            if current == 0 and not near_roll:
                # Curve shape filter: only trade aligned with regime
                if z < -self.cfg.entry_zscore:
                    # Spread too narrow → long (front cheap vs back)
                    if not self.cfg.use_curve_shape or sh >= 0:
                        current = 1
                elif z > self.cfg.entry_zscore:
                    # Spread too wide → short (front expensive vs back)
                    if not self.cfg.use_curve_shape or sh <= 0:
                        current = -1
            elif current == 1:
                if z >= -self.cfg.exit_zscore or z <= -self.cfg.stop_zscore or near_roll:
                    current = 0
            elif current == -1:
                if z <= self.cfg.exit_zscore or z >= self.cfg.stop_zscore or near_roll:
                    current = 0

            position.iloc[i] = current

        daily_pnl = position.shift(1) * spread.diff()

        self.results = pd.DataFrame({
            "front_Q":      self.front,
            "back_Q":       self.back,
            "cal":          self.cal if self.cal is not None else np.nan,
            "roll_spread":  spread,
            "roll_mean":    roll_mean,
            "zscore":       zscore,
            "curve_shape":  shape,
            "days_to_roll": dtr,
            "position":     position,
            "daily_pnl":    daily_pnl,
            "cum_pnl":      daily_pnl.cumsum(),
        })
        return self.results

    # ------------------------------------------------------------------
    # Roll calendar
    # ------------------------------------------------------------------

    def roll_calendar(self) -> pd.DataFrame:
        """
        Generate a roll calendar showing all quarterly roll dates
        in the dataset, with the spread level at each roll.
        """
        spread = self.compute_spread()
        records = []
        prev_q  = None
        for dt, val in spread.items():
            q = quarter_label(dt)
            if q != prev_q and prev_q is not None:
                records.append({
                    "roll_date":    dt,
                    "from_quarter": prev_q,
                    "to_quarter":   q,
                    "spread_at_roll": round(val, 2),
                })
            prev_q = q
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        pnl    = self.results["daily_pnl"].dropna()
        spread = self.results["roll_spread"].dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        dd     = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        return {
            "market":            self.cfg.market,
            "spread_mean":       round(spread.mean(), 2),
            "spread_std":        round(spread.std(), 2),
            "contango_days_pct": round((spread < 0).mean() * 100, 1),
            "backw_days_pct":    round((spread > 0).mean() * 100, 1),
            "total_pnl":         round(pnl.sum(), 2),
            "sharpe_ratio":      round(sharpe, 3),
            "max_drawdown":      round(dd, 2),
            "win_rate":          round((pnl > 0).mean(), 3),
            "n_long":            int((self.results["position"] == 1).sum()),
            "n_short":           int((self.results["position"] == -1).sum()),
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, figsize=(14, 12)) -> plt.Figure:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        df  = self.results
        fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
        fig.suptitle(f"Quarterly Roll Strategy — {self.cfg.market}",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["front_Q"], label="Front Q",  color="#1565c0", lw=1.0)
        ax.plot(df.index, df["back_Q"],  label="Back Q+1", color="#e65100", lw=1.0, ls="--")
        if not df["cal"].isna().all():
            ax.plot(df.index, df["cal"], label="Cal Year", color="#555", lw=0.7, ls=":")
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("Quarter Prices", fontsize=10)

        ax = axes[1]
        ax.plot(df.index, df["roll_spread"], color="#333", lw=0.9, label="Roll Spread (F-B)")
        ax.plot(df.index, df["roll_mean"],   color="#9c27b0", lw=0.8, ls="--", label="Rolling Mean")
        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.fill_between(df.index, df["roll_spread"], 0,
                        where=df["roll_spread"] > 0, color="#1565c0", alpha=0.15)
        ax.fill_between(df.index, df["roll_spread"], 0,
                        where=df["roll_spread"] < 0, color="#e65100", alpha=0.15)
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("Roll Spread", fontsize=10)

        ax = axes[2]
        ax.plot(df.index, df["days_to_roll"], color="#2e7d32", lw=0.8)
        ax.axhline(self.cfg.roll_window_days, color="red", lw=0.7, ls="--",
                   label=f"Roll window ({self.cfg.roll_window_days}d)")
        ax.legend(fontsize=8); ax.set_ylabel("Days"); ax.set_title("Days to Roll", fontsize=10)

        ax = axes[3]
        ax.plot(df.index, df["zscore"], color="#555", lw=0.8)
        ax.axhline(self.cfg.entry_zscore,  color="red",   lw=0.8, ls="--")
        ax.axhline(-self.cfg.entry_zscore, color="green", lw=0.8, ls="--")
        ax.axhline(0, color="black", lw=0.3)
        longs  = df[df["position"] == 1]
        shorts = df[df["position"] == -1]
        ax.scatter(longs.index,  longs["zscore"],  s=14, color="green", zorder=5, label="Long")
        ax.scatter(shorts.index, shorts["zscore"], s=14, color="red",   zorder=5, label="Short")
        ax.legend(fontsize=8); ax.set_ylabel("Z-score"); ax.set_title("Signal", fontsize=10)

        ax = axes[4]
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
    rng   = np.random.default_rng(55)
    n     = 750
    dates = pd.date_range("2021-01-01", periods=n, freq="B")

    base    = 80 + np.cumsum(rng.normal(0, 1.1, n))
    base    = np.clip(base, 30, 400)
    front_q = base + rng.normal(0, 2.5, n)
    back_q  = base - 2 + rng.normal(0, 2.0, n)   # slight contango
    cal     = base - 1 + rng.normal(0, 1.0, n)

    cfg = QuarterlyRollConfig(
        lookback=30, entry_zscore=1.7,
        use_roll_timing=True, use_curve_shape=True,
        market="Power DE",
    )
    strat = QuarterlyRollStrategy(
        front_quarter = pd.Series(front_q, index=dates),
        back_quarter  = pd.Series(back_q,  index=dates),
        cal_year      = pd.Series(cal,     index=dates),
        config=cfg,
    )

    results = strat.run()
    stats   = strat.summary()

    print("\n=== Quarterly Roll Strategy — Summary ===")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    print("\n  Roll Calendar (first 5 rolls):")
    print(strat.roll_calendar().head().to_string(index=False))

    fig = strat.plot()
    fig.savefig("quarterly_roll.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → quarterly_roll.png")
