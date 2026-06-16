"""
Supply & Demand Model — Net Load Forecasting
=============================================
Directional trading strategy driven by supply/demand balance analysis
for European power markets.

Market rationale:
    Power prices are ultimately set by the supply-demand balance.
    The key concept is NET LOAD:

        Net Load = Total Load - Variable Renewable Generation
                 = Residual demand that must be served by dispatchable plants

    When net load is HIGH  → expensive peaking plants set the price → buy power
    When net load is LOW   → cheap renewables surplus → sell power (or even negative prices)

    Key drivers:
        DEMAND SIDE:
            - Temperature (heating/cooling degree days)
            - Industrial production index
            - Day of week / hour of day (load profile)
            - Public holidays

        SUPPLY SIDE:
            - Wind generation forecast (anti-correlated with price)
            - Solar PV forecast (mid-day price suppression)
            - Hydro availability
            - Nuclear availability (especially France)
            - Thermal outages (unplanned)

Strategy:
    1. Forecast net load for the following day/week
    2. Map net load to expected price range via merit order lookup
    3. Generate directional signals (buy/sell Day-Ahead power)
    4. Size position based on confidence in forecast

Data sources:
    - ENTSO-E Transparency Platform: load, generation by source
    - National weather services / ECMWF forecasts
    - TSO generation forecasts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from scipy import stats as scipy_stats


@dataclass
class MarketParams:
    """Market-specific parameters."""
    zone: str                  = "DE_LU"
    peak_hours: tuple          = (8, 20)        # peak hours range
    base_load_mw: float        = 45_000         # typical base load [MW]
    max_renewable_mw: float    = 80_000         # installed renewable capacity [MW]
    price_floor: float         = -50.0          # minimum price [€/MWh]
    price_cap: float           = 3_000.0        # maximum price [€/MWh]


@dataclass
class SDModelConfig:
    """Configuration for the supply-demand model strategy."""
    lookback: int              = 30             # days for rolling calibration
    forecast_horizon: int      = 1              # days ahead to forecast
    entry_zscore: float        = 1.5
    exit_zscore: float         = 0.4
    stop_zscore: float         = 3.5
    use_temperature: bool      = True
    use_wind: bool             = True
    use_solar: bool            = True
    use_hydro: bool            = True


class NetLoadModel:
    """
    Computes and analyses net load time series.

    Net Load = Total Load - Wind - Solar - Run-of-river Hydro

    Parameters
    ----------
    load        : pd.Series   Total system load [MW].
    wind        : pd.Series   Wind generation [MW].
    solar       : pd.Series   Solar PV generation [MW].
    hydro_ror   : pd.Series   Run-of-river hydro [MW] (optional).
    temperature : pd.Series   Average temperature [°C] (optional).
    """

    def __init__(
        self,
        load: pd.Series,
        wind: pd.Series,
        solar: pd.Series,
        hydro_ror: Optional[pd.Series] = None,
        temperature: Optional[pd.Series] = None,
    ):
        self.load        = load.rename("load_mw")
        self.wind        = wind.rename("wind_mw")
        self.solar       = solar.rename("solar_mw")
        self.hydro_ror   = hydro_ror.rename("hydro_mw") if hydro_ror is not None else None
        self.temperature = temperature.rename("temperature") if temperature is not None else None

    def net_load(self) -> pd.Series:
        """Net load [MW]: load minus all variable renewables."""
        nl = self.load - self.wind - self.solar
        if self.hydro_ror is not None:
            nl -= self.hydro_ror
        nl.name = "net_load_mw"
        return nl.clip(lower=0)

    def renewable_penetration(self) -> pd.Series:
        """Fraction of load served by variable renewables [0–1]."""
        vre = self.wind + self.solar
        if self.hydro_ror is not None:
            vre += self.hydro_ror
        pen = vre / self.load.replace(0, np.nan)
        pen.name = "vre_penetration"
        return pen.clip(0, 1)

    def curtailment_risk(self) -> pd.Series:
        """
        Proxy for renewable curtailment risk.
        High when renewable generation exceeds minimum load.
        Returns fraction of excess [0+].
        """
        vre = self.wind + self.solar
        if self.hydro_ror is not None:
            vre += self.hydro_ror
        excess = (vre - self.load).clip(lower=0)
        risk   = excess / self.load.replace(0, np.nan)
        risk.name = "curtailment_risk"
        return risk

    def heating_cooling_degree_days(self, base_temp: float = 15.0) -> pd.DataFrame:
        """
        Compute Heating Degree Days (HDD) and Cooling Degree Days (CDD).
        HDD = max(base_temp - T, 0)
        CDD = max(T - base_temp, 0)
        """
        if self.temperature is None:
            raise ValueError("Temperature data required.")
        hdd = (base_temp - self.temperature).clip(lower=0).rename("HDD")
        cdd = (self.temperature - base_temp).clip(lower=0).rename("CDD")
        return pd.DataFrame({"HDD": hdd, "CDD": cdd})

    def seasonal_decomposition(self, series: pd.Series) -> pd.DataFrame:
        """Simple seasonal decomposition using rolling means."""
        trend    = series.rolling(30, center=True, min_periods=10).mean()
        seasonal = series.groupby(series.index.dayofyear).transform("mean")
        residual = series - trend - (seasonal - seasonal.mean())
        return pd.DataFrame({
            "original": series,
            "trend":    trend,
            "seasonal": seasonal,
            "residual": residual,
        })


class SupplyDemandStrategy:
    """
    Trading strategy based on net load forecast and supply-demand balance.

    Parameters
    ----------
    power_prices : pd.Series   Day-ahead power prices [€/MWh].
    net_load_mdl : NetLoadModel
    config       : SDModelConfig
    params       : MarketParams
    """

    def __init__(
        self,
        power_prices: pd.Series,
        net_load_model: NetLoadModel,
        config: Optional[SDModelConfig]  = None,
        params: Optional[MarketParams]   = None,
    ):
        self.prices = power_prices.rename("power_price")
        self.model  = net_load_model
        self.cfg    = config or SDModelConfig()
        self.params = params or MarketParams()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Price-netload relationship
    # ------------------------------------------------------------------

    def calibrate_price_netload(
        self, net_load: pd.Series
    ) -> Tuple[float, float, float]:
        """
        OLS regression: Price ~ alpha + beta * NetLoad + epsilon
        Returns (alpha, beta, r_squared).
        Higher net load → higher price (positive beta expected).
        """
        df = pd.DataFrame({"price": self.prices, "nl": net_load}).dropna()
        if len(df) < 10:
            return 0.0, 0.0, 0.0
        slope, intercept, r, _, _ = scipy_stats.linregress(df["nl"], df["price"])
        return round(intercept, 4), round(slope, 6), round(r**2, 4)

    def price_forecast(
        self, net_load: pd.Series, lookback: int = 60
    ) -> pd.Series:
        """
        Rolling price forecast based on net load.
        At each point, calibrate OLS on past `lookback` days,
        then forecast today's price from today's net load.
        """
        forecasts = pd.Series(np.nan, index=net_load.index, name="price_forecast")
        for i in range(lookback, len(net_load)):
            nl_window    = net_load.iloc[i - lookback: i]
            px_window    = self.prices.iloc[i - lookback: i]
            df_w         = pd.DataFrame({"px": px_window, "nl": nl_window}).dropna()
            if len(df_w) < 10:
                continue
            sl, ic, _, _, _ = scipy_stats.linregress(df_w["nl"], df_w["px"])
            forecasts.iloc[i] = ic + sl * net_load.iloc[i]
        return forecasts

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Generate trading signals based on:
        1. Net load z-score (high net load → buy signal)
        2. Price vs forecast deviation (price cheap vs model → buy)
        3. Renewable penetration regime filter
        """
        net_load   = self.model.net_load()
        vre_pen    = self.model.renewable_penetration()
        curt_risk  = self.model.curtailment_risk()

        # Net load z-score
        nl_mean    = net_load.rolling(self.cfg.lookback).mean()
        nl_std     = net_load.rolling(self.cfg.lookback).std()
        nl_zscore  = (net_load - nl_mean) / nl_std.replace(0, np.nan)

        # Price forecast signal
        px_forecast = self.price_forecast(net_load, lookback=self.cfg.lookback * 2)
        px_vs_model = self.prices - px_forecast   # positive = price above model
        px_roll_std = px_vs_model.rolling(self.cfg.lookback).std()
        px_zscore   = px_vs_model / px_roll_std.replace(0, np.nan)

        # Combined signal: net load high AND price cheap vs model = buy
        combined_z  = nl_zscore - px_zscore       # high NL + cheap price = large positive

        position = pd.Series(0.0, index=self.prices.index)
        current  = 0

        for i in range(self.cfg.lookback * 2, len(combined_z)):
            z    = combined_z.iloc[i]
            vre  = vre_pen.iloc[i]
            curt = curt_risk.iloc[i]
            if np.isnan(z):
                continue

            # Curtailment regime: avoid longs when renewables likely to cause negative prices
            high_vre = (vre > 0.70) or (curt > 0.10)

            if current == 0:
                if z > self.cfg.entry_zscore and not high_vre:
                    current = 1     # high net load, price cheap vs model → buy
                elif z < -self.cfg.entry_zscore:
                    current = -1    # low net load, price rich vs model → sell
            elif current == 1:
                if z < self.cfg.exit_zscore or z < -self.cfg.stop_zscore:
                    current = 0
            elif current == -1:
                if z > -self.cfg.exit_zscore or z > self.cfg.stop_zscore:
                    current = 0

            position.iloc[i] = current

        daily_pnl = position.shift(1) * self.prices.diff()

        self.results = pd.DataFrame({
            "power_price":    self.prices,
            "net_load":       net_load,
            "vre_pen":        vre_pen,
            "curt_risk":      curt_risk,
            "nl_zscore":      nl_zscore,
            "px_forecast":    px_forecast,
            "px_vs_model":    px_vs_model,
            "combined_z":     combined_z,
            "position":       position,
            "daily_pnl":      daily_pnl,
            "cum_pnl":        daily_pnl.cumsum(),
        })
        return self.results

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def price_netload_summary(self) -> dict:
        """Summary of the price-netload relationship."""
        nl = self.model.net_load()
        alpha, beta, r2 = self.calibrate_price_netload(nl)
        return {
            "zone":              self.params.zone,
            "price_nl_alpha":    alpha,
            "price_nl_beta_per_gw": round(beta * 1000, 4),
            "price_nl_r_squared": r2,
            "avg_net_load_mw":   round(nl.mean(), 0),
            "avg_vre_pen_pct":   round(self.model.renewable_penetration().mean() * 100, 1),
            "neg_price_days_pct":round((self.prices < 0).mean() * 100, 1),
        }

    def summary(self) -> dict:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        pnl    = self.results["daily_pnl"].dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        dd     = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        return {
            **self.price_netload_summary(),
            "total_pnl":    round(pnl.sum(), 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(dd, 2),
            "win_rate":     round((pnl > 0).mean(), 3),
            "n_long":       int((self.results["position"] == 1).sum()),
            "n_short":      int((self.results["position"] == -1).sum()),
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, figsize=(14, 14)) -> plt.Figure:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        df  = self.results
        fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
        fig.suptitle(f"Supply & Demand Model — {self.params.zone}",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["power_price"],  label="DA Price", color="#1565c0", lw=1.0)
        ax.plot(df.index, df["px_forecast"],  label="Model Forecast", color="#e65100",
                lw=0.9, ls="--")
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("Power Price vs Model", fontsize=10)

        ax = axes[1]
        ax.fill_between(df.index, df["net_load"] / 1000, 0, color="#37474f", alpha=0.5)
        ax.set_ylabel("GW"); ax.set_title("Net Load (Residual Demand)", fontsize=10)

        ax = axes[2]
        ax.fill_between(df.index, df["vre_pen"] * 100, 0,
                        color="#2e7d32", alpha=0.6, label="VRE Penetration")
        ax.axhline(70, color="red", lw=0.8, ls="--", label="70% curtailment risk")
        ax.legend(fontsize=8); ax.set_ylabel("%")
        ax.set_title("Renewable Penetration", fontsize=10)

        ax = axes[3]
        ax.plot(df.index, df["combined_z"], color="#555", lw=0.8)
        ax.axhline(self.cfg.entry_zscore,  color="green", lw=0.8, ls="--")
        ax.axhline(-self.cfg.entry_zscore, color="red",   lw=0.8, ls="--")
        ax.axhline(0, color="black", lw=0.3)
        longs  = df[df["position"] == 1]
        shorts = df[df["position"] == -1]
        ax.scatter(longs.index,  longs["combined_z"],  s=14, color="green", zorder=5, label="Long")
        ax.scatter(shorts.index, shorts["combined_z"], s=14, color="red",   zorder=5, label="Short")
        ax.legend(fontsize=8); ax.set_ylabel("Z-score")
        ax.set_title("Combined Signal (Net Load − Price Model)", fontsize=10)

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
    rng   = np.random.default_rng(17)
    n     = 800
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Seasonal load pattern
    doy   = np.array([d.dayofyear for d in dates])
    load  = 50000 + 15000 * np.cos(2*np.pi*(doy-15)/365) + rng.normal(0, 2000, n)
    wind  = np.clip(15000 + 8000 * np.cos(2*np.pi*(doy-355)/365) + rng.normal(0, 3000, n), 0, 60000)
    solar = np.clip(8000  - 6000 * np.cos(2*np.pi*(doy-15)/365)  + rng.normal(0, 1500, n), 0, 30000)
    hydro = np.clip(3000  + rng.normal(0, 500, n), 500, 8000)
    temp  = 10 - 12 * np.cos(2*np.pi*(doy-15)/365) + rng.normal(0, 3, n)

    net_l = np.clip(load - wind - solar - hydro, 5000, 70000)
    price = 30 + 0.0008 * net_l + rng.normal(0, 8, n)
    price = np.clip(price, -50, 300)

    mdl = NetLoadModel(
        load        = pd.Series(load,  index=dates),
        wind        = pd.Series(wind,  index=dates),
        solar       = pd.Series(solar, index=dates),
        hydro_ror   = pd.Series(hydro, index=dates),
        temperature = pd.Series(temp,  index=dates),
    )
    strat = SupplyDemandStrategy(
        power_prices   = pd.Series(price, index=dates),
        net_load_model = mdl,
        config=SDModelConfig(lookback=30),
        params=MarketParams(zone="DE_LU"),
    )

    results = strat.run()
    stats   = strat.summary()
    print("\n=== Supply & Demand Model — Summary ===")
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")

    fig = strat.plot()
    fig.savefig("supply_demand_model.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → supply_demand_model.png")
