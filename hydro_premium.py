"""
Hydro Reservoir Premium Strategy
==================================
Trades power prices based on hydro reservoir levels and their
impact on the electricity supply stack.

Market rationale:
    Hydroelectric generation is a zero-marginal-cost, flexible resource.
    Reservoir levels directly affect:

    1. SUPPLY AVAILABILITY:
       High reservoir → abundant cheap hydro → power price depressed
       Low reservoir  → scarce hydro → thermal plants set price → higher prices

    2. HYDRO PREMIUM:
       Markets price in expected future hydro availability.
       The "hydro premium" is the difference between current prices
       and what prices would be with average hydro conditions.

    3. SEASONALITY:
       Alpine reservoirs: fill in spring (snowmelt), drain in winter.
       Scandinavian hydro: dominant factor for Nordic power prices.
       Iberian hydro: significant in wet/dry years (Iberian Peninsula).

    4. MEAN REVERSION:
       Reservoir levels are bounded (0–100%) and exhibit strong
       seasonal mean reversion → predictable signal.

Key markets most affected:
    - Nordic (NO, SE, FI): 90%+ hydro penetration
    - Alpine (AT, CH): 60-70% hydro
    - Iberian (ES, PT): 15-25% hydro, high inter-year variability
    - French (FR): 10-15% hydro, important for peak demand

Strategy:
    1. Compare reservoir fill vs seasonal normal (hydro deviation)
    2. High deviation (over-filled) → short power (bearish supply)
    3. Low deviation (under-filled) → long power (bullish supply)
    4. Combine with inflow forecast for forward-looking signal
    5. Apply market-specific calibration

Data sources:
    - NVE (Norway): weekly hydro statistics
    - Nordpool: hydro balance
    - ENTSO-E: hydro storage levels
    - REE (Spain): hydro reservoir data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class HydroMarketParams:
    """Market-specific hydro parameters."""
    market: str                  = "Nordic"
    hydro_share: float           = 0.65      # fraction of generation from hydro
    reservoir_capacity_twh: float = 120.0    # total reservoir capacity [TWh]
    typical_fill_pct: float       = 65.0     # long-run average fill %
    fill_std_pct: float           = 12.0     # typical seasonal std dev %
    price_sensitivity: float      = 0.8      # €/MWh per % deviation from normal


@dataclass
class HydroConfig:
    """Strategy configuration."""
    lookback: int                = 30
    entry_zscore: float          = 1.5
    exit_zscore: float           = 0.4
    stop_zscore: float           = 3.5
    inflow_lookback: int         = 14        # days for inflow trend
    use_inflow: bool             = True
    use_temperature: bool        = True


class HydroReservoirModel:
    """
    Models hydro reservoir dynamics and computes price impact signals.

    Parameters
    ----------
    reservoir_fill_pct : pd.Series   Reservoir fill level [%].
    inflow_gwh_day     : pd.Series   Daily water inflow [GWh/day] (optional).
    temperature        : pd.Series   Average temperature [°C] (optional).
    params             : HydroMarketParams
    """

    def __init__(
        self,
        reservoir_fill_pct: pd.Series,
        inflow_gwh_day: Optional[pd.Series] = None,
        temperature: Optional[pd.Series]    = None,
        params: Optional[HydroMarketParams] = None,
    ):
        self.fill        = reservoir_fill_pct.rename("reservoir_fill_pct")
        self.inflow      = inflow_gwh_day.rename("inflow_gwh") if inflow_gwh_day is not None else None
        self.temperature = temperature.rename("temperature")   if temperature is not None else None
        self.params      = params or HydroMarketParams()

    def seasonal_normal(self) -> pd.Series:
        """
        Seasonal normal fill level: average fill for each week-of-year.
        """
        woy  = self.fill.index.isocalendar().week.astype(int)
        norm = self.fill.groupby(woy).transform("mean")
        norm.name = "seasonal_normal_pct"
        return norm

    def hydro_deviation(self) -> pd.Series:
        """
        Deviation from seasonal normal [percentage points].
        Positive = above normal (bearish power prices).
        Negative = below normal (bullish power prices).
        """
        dev = self.fill - self.seasonal_normal()
        dev.name = "hydro_deviation_pp"
        return dev

    def hydro_deviation_zscore(self, lookback: int = 52) -> pd.Series:
        """
        Z-score of hydro deviation over rolling window (weeks).
        """
        dev  = self.hydro_deviation()
        mean = dev.rolling(lookback).mean()
        std  = dev.rolling(lookback).std().replace(0, np.nan)
        z    = (dev - mean) / std
        z.name = "hydro_dev_zscore"
        return z

    def price_impact_estimate(self) -> pd.Series:
        """
        Rough estimate of hydro premium/discount on power price.
        Based on hydro deviation × price sensitivity parameter.
        Negative deviation (low hydro) → positive price impact (premium).
        """
        dev    = self.hydro_deviation()
        impact = -dev * self.params.price_sensitivity
        impact.name = "hydro_price_impact_eur_mwh"
        return impact

    def inflow_trend(self, lookback: int = 14) -> Optional[pd.Series]:
        """
        Rolling trend of water inflows.
        Rising inflows → bearish (more generation available soon).
        Falling inflows → bullish.
        Returns normalised trend [-1, +1].
        """
        if self.inflow is None:
            return None
        inflow_ma   = self.inflow.rolling(lookback).mean()
        inflow_lag  = self.inflow.rolling(lookback * 2).mean()
        trend = (inflow_ma - inflow_lag) / inflow_lag.replace(0, np.nan)
        trend = trend.clip(-1, 1)
        trend.name = "inflow_trend"
        return trend

    def snowmelt_signal(self) -> pd.Series:
        """
        Snowmelt season proxy: spring months (Apr-Jun) in cold climates.
        During snowmelt: rapid inflow increase → bearish hydro premium.
        Returns 1 during snowmelt season, 0 otherwise.
        """
        months = self.fill.index.month
        snowmelt = pd.Series(0, index=self.fill.index, name="snowmelt_season")
        snowmelt[(months >= 4) & (months <= 6)] = 1
        return snowmelt

    def dry_year_indicator(self, threshold_pct: float = -15.0) -> pd.Series:
        """
        Identifies dry years where reservoir fill is persistently below normal.
        Dry years cause sustained power price premiums.
        Returns rolling 90-day average of hydro deviation.
        """
        dev = self.hydro_deviation()
        dry = dev.rolling(90, min_periods=30).mean()
        dry.name = "rolling_90d_deviation"
        return dry


class HydroPremiumStrategy:
    """
    Power trading strategy based on hydro reservoir premium signal.

    Parameters
    ----------
    power_prices      : pd.Series       Power prices [€/MWh].
    hydro_model       : HydroReservoirModel
    config            : HydroConfig
    """

    def __init__(
        self,
        power_prices: pd.Series,
        hydro_model: HydroReservoirModel,
        config: Optional[HydroConfig] = None,
    ):
        self.prices = power_prices.rename("power_price")
        self.model  = hydro_model
        self.cfg    = config or HydroConfig()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------

    def _build_signal(self) -> pd.Series:
        """
        Combine hydro deviation z-score, inflow trend, and snowmelt.
        Positive combined signal → long power (low hydro).
        Negative combined signal → short power (high hydro).
        """
        dev_z      = self.model.hydro_deviation_zscore(lookback=self.cfg.lookback)
        inflow_tr  = self.model.inflow_trend(self.cfg.inflow_lookback) if self.cfg.use_inflow else None
        snowmelt   = self.model.snowmelt_signal()

        # Base signal: negative deviation z-score = long power
        combined = -dev_z.copy()

        # Inflow trend overlay: rising inflow → subtract from signal (bearish)
        if inflow_tr is not None:
            combined -= inflow_tr

        # Snowmelt seasonal overlay: during snowmelt, lean short
        combined -= snowmelt * 0.5

        combined.name = "combined_signal"
        return combined

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Generate directional power price signals from hydro analysis."""
        dev_z     = self.model.hydro_deviation_zscore(lookback=self.cfg.lookback)
        dev       = self.model.hydro_deviation()
        impact    = self.model.price_impact_estimate()
        inflow_tr = self.model.inflow_trend(self.cfg.inflow_lookback)
        snowmelt  = self.model.snowmelt_signal()
        combined  = self._build_signal()

        position = pd.Series(0.0, index=self.prices.index)
        current  = 0

        for i in range(self.cfg.lookback, len(combined)):
            c = combined.iloc[i]
            if np.isnan(c):
                continue
            if current == 0:
                if c > self.cfg.entry_zscore:
                    current =  1    # low hydro → long power
                elif c < -self.cfg.entry_zscore:
                    current = -1    # high hydro → short power
            elif current == 1:
                if c < self.cfg.exit_zscore or c < -self.cfg.stop_zscore:
                    current = 0
            elif current == -1:
                if c > -self.cfg.exit_zscore or c > self.cfg.stop_zscore:
                    current = 0
            position.iloc[i] = current

        daily_pnl = position.shift(1) * self.prices.diff()

        self.results = pd.DataFrame({
            "power_price":     self.prices,
            "reservoir_fill":  self.model.fill,
            "seasonal_normal": self.model.seasonal_normal(),
            "hydro_deviation": dev,
            "dev_zscore":      dev_z,
            "price_impact":    impact,
            "inflow_trend":    inflow_tr if inflow_tr is not None else np.nan,
            "snowmelt":        snowmelt,
            "combined":        combined,
            "position":        position,
            "daily_pnl":       daily_pnl,
            "cum_pnl":         daily_pnl.cumsum(),
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
        params = self.model.params
        return {
            "market":               params.market,
            "hydro_share_pct":      round(params.hydro_share * 100, 1),
            "avg_fill_pct":         round(self.model.fill.mean(), 1),
            "avg_deviation_pp":     round(self.model.hydro_deviation().mean(), 2),
            "dry_year_days_pct":    round((self.model.hydro_deviation() < -10).mean() * 100, 1),
            "wet_year_days_pct":    round((self.model.hydro_deviation() >  10).mean() * 100, 1),
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
        fig.suptitle(f"Hydro Premium Strategy — {self.model.params.market}",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["power_price"], color="#1565c0", lw=1.0, label="Power Price")
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh")
        ax.set_title("Power Price", fontsize=10)

        ax = axes[1]
        ax.plot(df.index, df["reservoir_fill"],  color="#0288d1", lw=1.0, label="Fill %")
        ax.plot(df.index, df["seasonal_normal"], color="#888",    lw=0.9, ls="--", label="Seasonal Normal")
        ax.fill_between(df.index, df["reservoir_fill"], df["seasonal_normal"],
                        where=df["reservoir_fill"] > df["seasonal_normal"],
                        color="blue", alpha=0.2, label="Above normal (bearish)")
        ax.fill_between(df.index, df["reservoir_fill"], df["seasonal_normal"],
                        where=df["reservoir_fill"] < df["seasonal_normal"],
                        color="red",  alpha=0.2, label="Below normal (bullish)")
        ax.legend(fontsize=8); ax.set_ylabel("%")
        ax.set_title("Reservoir Fill vs Seasonal Normal", fontsize=10)

        ax = axes[2]
        ax.bar(df.index, df["hydro_deviation"], color=np.where(df["hydro_deviation"] < 0, "#c62828", "#1565c0"),
               width=1, alpha=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("pp deviation"); ax.set_title("Hydro Deviation from Seasonal Normal", fontsize=10)

        ax = axes[3]
        ax.plot(df.index, df["combined"], color="#555", lw=0.8)
        ax.axhline(self.cfg.entry_zscore,  color="green", lw=0.8, ls="--")
        ax.axhline(-self.cfg.entry_zscore, color="red",   lw=0.8, ls="--")
        ax.axhline(0, color="black", lw=0.3)
        longs  = df[df["position"] == 1]
        shorts = df[df["position"] == -1]
        ax.scatter(longs.index,  longs["combined"],  s=14, color="green", zorder=5, label="Long power")
        ax.scatter(shorts.index, shorts["combined"], s=14, color="red",   zorder=5, label="Short power")
        ax.legend(fontsize=8); ax.set_ylabel("Signal")
        ax.set_title("Combined Hydro Signal", fontsize=10)

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
    rng   = np.random.default_rng(77)
    n     = 800
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    doy   = np.array([d.dayofyear for d in dates])

    # Nordic-style reservoir: fills spring, drains winter
    fill   = np.clip(65 + 30*np.sin(2*np.pi*(doy-120)/365) + rng.normal(0, 5, n), 5, 100)
    inflow = np.clip(200 + 150*np.sin(2*np.pi*(doy-100)/365) + rng.normal(0, 30, n), 10, 600)
    temp   = np.clip(5 - 20*np.cos(2*np.pi*(doy-15)/365) + rng.normal(0, 3, n), -20, 35)

    # Power price: inversely correlated with hydro fill
    power  = np.clip(60 - 0.4*(fill-65) + rng.normal(0, 6, n), 5, 250)

    params = HydroMarketParams(market="Nordic", hydro_share=0.65,
                               reservoir_capacity_twh=120, price_sensitivity=0.6)
    cfg    = HydroConfig(lookback=30, entry_zscore=1.5, use_inflow=True)

    mdl   = HydroReservoirModel(
        reservoir_fill_pct = pd.Series(fill,   index=dates),
        inflow_gwh_day     = pd.Series(inflow, index=dates),
        temperature        = pd.Series(temp,   index=dates),
        params=params,
    )
    strat = HydroPremiumStrategy(
        power_prices = pd.Series(power, index=dates),
        hydro_model  = mdl,
        config=cfg,
    )

    results = strat.run()
    stats   = strat.summary()
    print("\n=== Hydro Premium Strategy — Summary ===")
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")

    fig = strat.plot()
    fig.savefig("hydro_premium.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → hydro_premium.png")
