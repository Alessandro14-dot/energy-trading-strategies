"""
Gas Storage Valuation & Trading Strategy
=========================================
Models the optionality value of natural gas storage facilities
and generates trading signals based on storage economics.

Market rationale:
    A gas storage facility gives its owner the right (but not obligation)
    to inject gas when prices are low and withdraw when prices are high.
    This optionality has both INTRINSIC and EXTRINSIC value.

    INTRINSIC VALUE:
        The locked-in profit from the current forward curve shape.
        Inject when summer (cheap) and withdraw in winter (expensive).
        = Sum of max(0, P_winter - P_summer - costs) across all months

    EXTRINSIC VALUE (time value):
        The additional value from future price uncertainty.
        Even a flat forward curve has extrinsic value because
        prices may move favorably before injection/withdrawal.
        Captured via options or dynamic re-optimization.

    Key storage parameters:
        Working volume  : total gas that can be stored [GWh or mcm]
        Max inject rate : maximum daily injection [GWh/day]
        Max withdraw rate: maximum daily withdrawal [GWh/day]
        Injection cost  : €/MWh (compression, fuel, losses)
        Withdrawal cost : €/MWh

    Typical European gas storage:
        Total EU working volume: ~1,100 TWh
        Seasonal fill pattern: inject Apr-Sep, withdraw Oct-Mar
        AGSI+ tracks EU storage levels daily

Strategy signals:
    1. Storage vs seasonal average → over/under-stored
    2. Injection/withdrawal economics (spread vs cost)
    3. Days to winter signal (urgency to fill)
    4. Storage fair value vs market TTF spread

Data sources:
    AGSI+ (GIE): https://agsi.gie.eu  (free API)
    entsoe-py for power markets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ─── Storage facility parameters ─────────────────────────────────────────────

@dataclass
class StorageFacility:
    """Physical parameters of a gas storage facility."""
    name: str                   = "Generic_EU_Storage"
    working_volume_gwh: float   = 5_000.0       # total working volume [GWh]
    max_inject_gwh_day: float   = 100.0         # max daily injection [GWh/day]
    max_withdraw_gwh_day: float = 150.0         # max daily withdrawal [GWh/day]
    injection_cost: float       = 0.30          # €/MWh injected
    withdrawal_cost: float      = 0.20          # €/MWh withdrawn
    cycling_cost: float         = 0.05          # €/MWh (cushion gas, O&M)
    min_fill_pct: float         = 0.10          # minimum fill level (cushion gas)
    max_fill_pct: float         = 1.00          # maximum fill level

    @property
    def round_trip_cost(self) -> float:
        return self.injection_cost + self.withdrawal_cost + self.cycling_cost


@dataclass
class StorageConfig:
    """Strategy configuration."""
    lookback: int               = 30
    entry_zscore: float         = 1.6
    exit_zscore: float          = 0.4
    stop_zscore: float          = 3.5
    winter_months: tuple        = (10, 11, 12, 1, 2, 3)
    summer_months: tuple        = (4, 5, 6, 7, 8, 9)
    low_storage_threshold: float  = 35.0        # % fill → bullish gas
    high_storage_threshold: float = 85.0        # % fill → bearish gas


# ─── Storage economics ────────────────────────────────────────────────────────

class StorageValuation:
    """
    Computes intrinsic value of storage from forward curve.

    Parameters
    ----------
    summer_price  : float or pd.Series   Summer forward price [€/MWh].
    winter_price  : float or pd.Series   Winter forward price [€/MWh].
    facility      : StorageFacility
    """

    def __init__(
        self,
        summer_price,
        winter_price,
        facility: Optional[StorageFacility] = None,
    ):
        self.summer   = summer_price
        self.winter   = winter_price
        self.facility = facility or StorageFacility()

    def intrinsic_value(self):
        """
        Intrinsic value per MWh stored.
        IV = max(0, P_winter - P_summer - round_trip_cost)
        """
        spread = self.winter - self.summer
        iv     = np.maximum(0, spread - self.facility.round_trip_cost)
        return iv

    def break_even_spread(self) -> float:
        """Minimum W/S spread needed to cover round-trip cost."""
        return self.facility.round_trip_cost

    def max_cycle_value(self) -> float:
        """
        Maximum total value of one full storage cycle [€].
        = intrinsic_value * working_volume
        """
        iv = self.intrinsic_value()
        if hasattr(iv, "__len__"):
            iv = float(np.mean(iv))
        return iv * self.facility.working_volume_gwh * 1000  # GWh → MWh


# ─── Storage trading strategy ─────────────────────────────────────────────────

class StorageTradingStrategy:
    """
    Trading strategy based on gas storage levels and economics.

    Signal logic:
        - Below-seasonal storage + economic to inject → bullish gas (buy TTF)
        - Above-seasonal storage + uneconomic to inject → bearish gas (sell TTF)
        - Days-to-winter urgency overlay

    Parameters
    ----------
    gas_prices      : pd.Series   Spot/prompt gas prices [€/MWh].
    storage_fill_pct: pd.Series   EU or country storage fill level [%].
    summer_forward  : pd.Series   Summer strip forward price [€/MWh].
    winter_forward  : pd.Series   Winter strip forward price [€/MWh].
    facility        : StorageFacility
    config          : StorageConfig
    """

    def __init__(
        self,
        gas_prices: pd.Series,
        storage_fill_pct: pd.Series,
        summer_forward: pd.Series,
        winter_forward: pd.Series,
        facility: Optional[StorageFacility] = None,
        config: Optional[StorageConfig]     = None,
    ):
        self.gas      = gas_prices.rename("gas_price")
        self.fill     = storage_fill_pct.rename("storage_fill_pct")
        self.summer   = summer_forward.rename("summer_fwd")
        self.winter   = winter_forward.rename("winter_fwd")
        self.facility = facility or StorageFacility()
        self.cfg      = config   or StorageConfig()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Storage analytics
    # ------------------------------------------------------------------

    def seasonal_fill_norm(self) -> pd.Series:
        """
        Normalise fill level vs seasonal average.
        > 0: more filled than average for this time of year
        < 0: less filled than average for this time of year
        Returns z-score of fill vs seasonal mean.
        """
        doy  = self.fill.index.dayofyear
        seas_mean = self.fill.groupby(doy).transform("mean")
        seas_std  = self.fill.groupby(doy).transform("std").replace(0, np.nan)
        norm = (self.fill - seas_mean) / seas_std
        norm.name = "fill_seasonal_zscore"
        return norm

    def injection_economics(self) -> pd.Series:
        """
        Net economics of injecting today (summer) for winter withdrawal.
        = Winter_forward - Summer_forward - round_trip_cost
        Positive → economic to inject.
        """
        econ = self.winter - self.summer - self.facility.round_trip_cost
        econ.name = "injection_economics"
        return econ

    def days_to_winter(self) -> pd.Series:
        """
        Days until start of winter withdrawal season (Oct 1).
        Negative during winter (already withdrawing).
        """
        def _dtw(d: pd.Timestamp) -> int:
            target = pd.Timestamp(year=d.year, month=10, day=1)
            if d >= target:
                target = pd.Timestamp(year=d.year + 1, month=10, day=1)
            return (target - d).days
        dtw = pd.Series([_dtw(d) for d in self.fill.index],
                        index=self.fill.index, name="days_to_winter")
        return dtw

    def storage_urgency(self) -> pd.Series:
        """
        Urgency-to-fill score combining fill level and days to winter.
        High urgency → bullish gas (demand for injection).
        Score in [-1, +1].
        """
        dtw  = self.days_to_winter()
        fill = self.fill

        # Urgency rises as storage is low AND winter approaches
        urgency = pd.Series(0.0, index=fill.index, name="urgency")
        for i in range(len(fill)):
            f = fill.iloc[i]
            d = dtw.iloc[i]
            if d <= 0:
                urgency.iloc[i] = 0.0   # winter: no injection urgency
                continue
            # Target fill at winter start = 90%
            gap = max(0, 90 - f)        # how far below 90%
            # Normalise: if 180 days to go and 50% gap → medium urgency
            urg = min(1.0, (gap / 50) * (180 / max(d, 1)))
            urgency.iloc[i] = urg
        return urgency

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Generate gas price directional signals from storage analysis.
        +1 = long gas (bullish: low storage / injection urgency)
        -1 = short gas (bearish: high storage / withdrawal pressure)
        """
        fill_norm  = self.seasonal_fill_norm()
        inj_econ   = self.injection_economics()
        urgency    = self.storage_urgency()

        # Rolling z-score on gas price
        roll_mean  = self.gas.rolling(self.cfg.lookback).mean()
        roll_std   = self.gas.rolling(self.cfg.lookback).std()
        gas_zscore = (self.gas - roll_mean) / roll_std.replace(0, np.nan)

        # Combined signal score:
        #   -fill_norm  (low storage = bullish)
        #   +inj_econ   (economic to inject = bullish near-term)
        #   +urgency    (urgency to fill = bullish)
        inj_econ_z = (inj_econ - inj_econ.rolling(self.cfg.lookback).mean()) / \
                      inj_econ.rolling(self.cfg.lookback).std().replace(0, np.nan)

        combined = -fill_norm + inj_econ_z.fillna(0) + urgency * 2

        position = pd.Series(0.0, index=self.gas.index)
        current  = 0

        for i in range(self.cfg.lookback, len(combined)):
            c = combined.iloc[i]
            f = self.fill.iloc[i]
            if np.isnan(c):
                continue

            if current == 0:
                if c > self.cfg.entry_zscore and f < self.cfg.high_storage_threshold:
                    current =  1    # bullish gas
                elif c < -self.cfg.entry_zscore and f > self.cfg.low_storage_threshold:
                    current = -1    # bearish gas
            elif current == 1:
                if c < self.cfg.exit_zscore or c < -self.cfg.stop_zscore:
                    current = 0
            elif current == -1:
                if c > -self.cfg.exit_zscore or c > self.cfg.stop_zscore:
                    current = 0

            position.iloc[i] = current

        daily_pnl = position.shift(1) * self.gas.diff()

        self.results = pd.DataFrame({
            "gas_price":      self.gas,
            "storage_fill":   self.fill,
            "summer_fwd":     self.summer,
            "winter_fwd":     self.winter,
            "inj_economics":  inj_econ,
            "fill_norm":      fill_norm,
            "urgency":        urgency,
            "combined":       combined,
            "position":       position,
            "daily_pnl":      daily_pnl,
            "cum_pnl":        daily_pnl.cumsum(),
        })
        return self.results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        pnl    = self.results["daily_pnl"].dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        dd     = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        val    = StorageValuation(self.summer.mean(), self.winter.mean(), self.facility)
        return {
            "facility":             self.facility.name,
            "avg_storage_fill_pct": round(self.fill.mean(), 1),
            "inj_economic_pct":     round((self.results["inj_economics"] > 0).mean() * 100, 1),
            "breakeven_spread":     round(self.facility.round_trip_cost, 2),
            "avg_intrinsic_value":  round(float(val.intrinsic_value()), 2),
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

    def plot(self, figsize=(14, 14)) -> plt.Figure:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        df  = self.results
        fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
        fig.suptitle(f"Gas Storage Strategy — {self.facility.name}",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["gas_price"], color="#e65100", lw=1.0, label="Gas Price TTF")
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh")
        ax.set_title("Gas Price", fontsize=10)

        ax = axes[1]
        ax.fill_between(df.index, df["storage_fill"], 0, color="#1565c0", alpha=0.5)
        ax.axhline(self.cfg.high_storage_threshold, color="red",
                   lw=0.9, ls="--", label=f"High ({self.cfg.high_storage_threshold}%)")
        ax.axhline(self.cfg.low_storage_threshold,  color="green",
                   lw=0.9, ls="--", label=f"Low ({self.cfg.low_storage_threshold}%)")
        ax.legend(fontsize=8); ax.set_ylabel("%")
        ax.set_title("Storage Fill Level", fontsize=10)

        ax = axes[2]
        ax.plot(df.index, df["inj_economics"], color="#2e7d32", lw=0.9,
                label="Injection Economics (W-S spread - cost)")
        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.fill_between(df.index, df["inj_economics"], 0,
                        where=df["inj_economics"] > 0, color="green", alpha=0.3,
                        label="Economic to inject")
        ax.fill_between(df.index, df["inj_economics"], 0,
                        where=df["inj_economics"] < 0, color="red", alpha=0.3,
                        label="Uneconomic")
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh")
        ax.set_title("Injection Economics", fontsize=10)

        ax = axes[3]
        ax.plot(df.index, df["combined"], color="#555", lw=0.8)
        ax.axhline(self.cfg.entry_zscore,  color="green", lw=0.8, ls="--")
        ax.axhline(-self.cfg.entry_zscore, color="red",   lw=0.8, ls="--")
        ax.axhline(0, color="black", lw=0.3)
        longs  = df[df["position"] == 1]
        shorts = df[df["position"] == -1]
        ax.scatter(longs.index,  longs["combined"],  s=14, color="green", zorder=5, label="Long gas")
        ax.scatter(shorts.index, shorts["combined"], s=14, color="red",   zorder=5, label="Short gas")
        ax.legend(fontsize=8); ax.set_ylabel("Score")
        ax.set_title("Combined Storage Signal", fontsize=10)

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
    rng   = np.random.default_rng(44)
    n     = 800
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    doy   = np.array([d.dayofyear for d in dates])

    # Seasonal gas price
    gas   = np.clip(45 + 20*np.cos(2*np.pi*(doy-15)/365) +
                    np.cumsum(rng.normal(0, 1.2, n)), 10, 300)

    # Seasonal storage fill (inject Apr-Sep, drain Oct-Mar)
    fill  = np.clip(55 + 35*np.sin(2*np.pi*(doy-90)/365) +
                    rng.normal(0, 4, n), 5, 100)

    # Summer / winter forwards
    summer_fwd = np.clip(gas - 5 + rng.normal(0, 2, n), 10, 250)
    winter_fwd = np.clip(gas + 8 + rng.normal(0, 2, n), 10, 320)

    facility = StorageFacility(name="TTF_Storage_Generic",
                               working_volume_gwh=3000,
                               injection_cost=0.30, withdrawal_cost=0.20)
    cfg      = StorageConfig(lookback=30)

    strat = StorageTradingStrategy(
        gas_prices      = pd.Series(gas,        index=dates),
        storage_fill_pct= pd.Series(fill,       index=dates),
        summer_forward  = pd.Series(summer_fwd, index=dates),
        winter_forward  = pd.Series(winter_fwd, index=dates),
        facility=facility, config=cfg,
    )

    results = strat.run()
    stats   = strat.summary()
    print("\n=== Gas Storage Strategy — Summary ===")
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")

    fig = strat.plot()
    fig.savefig("storage_valuation.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → storage_valuation.png")
