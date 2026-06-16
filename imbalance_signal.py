"""
TSO Imbalance Price Signal Strategy
=====================================
Trades power based on forecasting TSO (Transmission System Operator)
imbalance settlement prices and exploiting their relationship with
intraday and balancing market prices.

Market background:
    Every power system has a real-time balancing mechanism.
    After gate closure, TSOs must balance supply and demand in real-time
    using balancing energy from activated reserves.

    Key concepts:

    IMBALANCE:    Difference between scheduled and actual generation/load.
                  Positive imbalance (system long) → TSO needs to absorb surplus
                  Negative imbalance (system short) → TSO needs to supply deficit

    SETTLEMENT:   Market participants are settled for their imbalance
                  at the imbalance settlement price (ISP).
                  ISP reflects the cost of TSO balancing actions.

    European balancing markets:
        Germany (REGELLEISTUNG.NET):
            - aFRR (automatic Frequency Restoration Reserve)
            - mFRR (manual Frequency Restoration Reserve)
            - Imbalance price = marginal balancing cost
            - Published ~30 min after each 15-min period

        Netherlands (TenneT NL):
            - Onbalans prijs published every 15 minutes
            - One-price settlement system

        Belgium (Elia):
            - NRV (Net Regulation Volume): system position
            - Published near real-time via API

        Italy (Terna - MSD):
            - Mercato del Servizio di Dispacciamento
            - Uplift/ancillary cost allocation

    Imbalance price predictors:
        1. System position (NRV): sign and magnitude of system imbalance
        2. Reserve activation: which balancing products are activated
        3. Renewable forecast error: wind/solar surprise
        4. Temperature deviation: unexpected demand changes
        5. Cross-border flows: congestion and import/export position

Strategy:
    Predict the sign and magnitude of the imbalance price.
    If imbalance price expected HIGH → be long (under-schedule, let TSO buy from you)
    If imbalance price expected LOW  → be short (over-schedule, sell back cheaply)

    Instruments:
        - Continuous intraday market (adjust schedules pre-gate)
        - Balancing market (direct bid to TSO, if licensed)
        - Portfolio rebalancing (shift between generation units)

    IMPORTANT NOTE:
        Direct participation in balancing markets requires TSO qualification.
        This strategy models the intraday hedge as proxy for balancing alpha.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Literal


@dataclass
class BalancingMarketParams:
    """Market-specific balancing parameters."""
    country: str                = "DE"
    settlement_period_min: int  = 15          # settlement period [minutes]
    gate_closure_min: int       = 45          # minutes before delivery
    max_imbalance_eur_mwh: float= 2000.0      # regulatory price cap
    min_imbalance_eur_mwh: float= -500.0      # negative prices allowed
    publication_lag_min: int    = 30          # lag between period end and price publication


@dataclass
class ImbalanceConfig:
    """Strategy configuration."""
    lookback: int               = 20          # periods for rolling stats
    entry_zscore: float         = 1.5
    exit_zscore: float          = 0.4
    stop_zscore: float          = 4.0
    position_size_mw: float     = 10.0       # MW per signal
    use_nrv: bool               = True        # use system position signal
    use_vre_surprise: bool      = True        # use renewable surprise signal
    use_price_momentum: bool    = True        # use intraday price momentum


class ImbalancePriceModel:
    """
    Model to forecast TSO imbalance settlement price direction.

    Key predictors:
        1. Net Regulation Volume (NRV): system position [MW]
           Negative NRV → system short → high imbalance price
           Positive NRV → system long  → low imbalance price

        2. Renewable surprise: actual - forecast [MW]
           Positive surprise (more VRE) → system long → low ISP
           Negative surprise (less VRE) → system short → high ISP

        3. Load surprise: actual - forecast [MW]
           Positive surprise (more load) → system short → high ISP

        4. Intraday price momentum: trend in ID prices
           Rising ID → bullish → higher ISP expected

    Parameters
    ----------
    imbalance_prices : pd.Series   Historical ISP [€/MWh], 15-min frequency.
    nrv              : pd.Series   Net Regulation Volume [MW].
    wind_surprise    : pd.Series   Wind generation surprise [MW].
    solar_surprise   : pd.Series   Solar generation surprise [MW].
    load_surprise    : pd.Series   Load surprise [MW].
    id_prices        : pd.Series   Intraday continuous prices [€/MWh].
    """

    def __init__(
        self,
        imbalance_prices: pd.Series,
        nrv: Optional[pd.Series]           = None,
        wind_surprise: Optional[pd.Series] = None,
        solar_surprise: Optional[pd.Series]= None,
        load_surprise: Optional[pd.Series] = None,
        id_prices: Optional[pd.Series]     = None,
        params: Optional[BalancingMarketParams] = None,
    ):
        self.isp      = imbalance_prices.rename("isp")
        self.nrv      = nrv
        self.wind_s   = wind_surprise
        self.solar_s  = solar_surprise
        self.load_s   = load_surprise
        self.id_px    = id_prices
        self.params   = params or BalancingMarketParams()

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def nrv_signal(self) -> Optional[pd.Series]:
        """
        NRV-based directional signal.
        Short system (negative NRV) → bullish ISP.
        """
        if self.nrv is None:
            return None
        # Normalise
        roll_std = self.nrv.rolling(96).std().replace(0, np.nan)   # 96 × 15min = 1 day
        sig      = -self.nrv / roll_std    # negative NRV → positive signal
        sig      = sig.clip(-3, 3) / 3
        sig.name = "nrv_signal"
        return sig

    def vre_surprise_signal(self) -> Optional[pd.Series]:
        """
        VRE surprise signal (negative surprise → bullish ISP).
        """
        vre = None
        if self.wind_s is not None:
            vre = self.wind_s.copy()
        if self.solar_s is not None:
            vre = vre.add(self.solar_s, fill_value=0) if vre is not None else self.solar_s.copy()
        if vre is None:
            return None
        roll_std = vre.rolling(96).std().replace(0, np.nan)
        sig      = -vre / roll_std   # more VRE than expected → bearish ISP
        sig      = sig.clip(-3, 3) / 3
        sig.name = "vre_surprise_signal"
        return sig

    def load_surprise_signal(self) -> Optional[pd.Series]:
        """Higher than expected load → bullish ISP."""
        if self.load_s is None:
            return None
        roll_std = self.load_s.rolling(96).std().replace(0, np.nan)
        sig      = self.load_s / roll_std
        sig      = sig.clip(-3, 3) / 3
        sig.name = "load_surprise_signal"
        return sig

    def id_momentum_signal(self, lookback: int = 12) -> Optional[pd.Series]:
        """
        Intraday price momentum: rising ID prices → bullish ISP.
        lookback in periods (e.g. 12 × 15min = 3 hours).
        """
        if self.id_px is None:
            return None
        mom = self.id_px.diff(lookback)
        roll_std = mom.rolling(lookback * 8).std().replace(0, np.nan)
        sig      = mom / roll_std
        sig      = sig.clip(-3, 3) / 3
        sig.name = "id_momentum_signal"
        return sig

    def composite_signal(self, weights: Dict[str, float] = None) -> pd.Series:
        """
        Weighted combination of all sub-signals.
        Returns composite signal in [-1, +1].
        """
        if weights is None:
            weights = {"nrv": 0.40, "vre": 0.30, "load": 0.15, "id_mom": 0.15}

        signals = {
            "nrv":    self.nrv_signal(),
            "vre":    self.vre_surprise_signal(),
            "load":   self.load_surprise_signal(),
            "id_mom": self.id_momentum_signal(),
        }

        composite = pd.Series(0.0, index=self.isp.index)
        total_w   = 0.0
        for key, sig in signals.items():
            w = weights.get(key, 0.0)
            if sig is not None and w > 0:
                aligned = sig.reindex(self.isp.index).fillna(0)
                composite += w * aligned
                total_w   += w

        if total_w > 0:
            composite /= total_w
        composite.name = "composite_signal"
        return composite.clip(-1, 1)

    # ------------------------------------------------------------------
    # ISP predictability analysis
    # ------------------------------------------------------------------

    def isp_statistics(self) -> dict:
        """Summary statistics of the imbalance settlement price."""
        isp = self.isp.dropna()
        return {
            "avg_isp":          round(isp.mean(), 2),
            "std_isp":          round(isp.std(), 2),
            "min_isp":          round(isp.min(), 2),
            "max_isp":          round(isp.max(), 2),
            "pct_above_100":    round((isp > 100).mean() * 100, 1),
            "pct_below_0":      round((isp < 0).mean() * 100, 1),
            "autocorr_lag1":    round(isp.autocorr(lag=1), 4),
            "autocorr_lag4":    round(isp.autocorr(lag=4), 4),   # 1 hour
        }

    def signal_isp_correlation(self) -> pd.DataFrame:
        """Correlations between sub-signals and next-period ISP."""
        isp_next = self.isp.shift(-1)    # next period ISP
        signals  = {
            "nrv_signal":    self.nrv_signal(),
            "vre_signal":    self.vre_surprise_signal(),
            "load_signal":   self.load_surprise_signal(),
            "id_mom_signal": self.id_momentum_signal(),
            "composite":     self.composite_signal(),
        }
        corrs = {}
        for name, sig in signals.items():
            if sig is not None:
                df_tmp = pd.concat([sig, isp_next], axis=1).dropna()
                if len(df_tmp) > 10:
                    corrs[name] = round(df_tmp.iloc[:, 0].corr(df_tmp.iloc[:, 1]), 4)
        return pd.Series(corrs, name="correlation_with_next_ISP")


class ImbalanceSignalStrategy:
    """
    Trading strategy based on imbalance price signal.

    Trades the intraday continuous market to position ahead of
    expected high/low imbalance settlement periods.

    Parameters
    ----------
    isp_model  : ImbalancePriceModel
    config     : ImbalanceConfig
    """

    def __init__(
        self,
        isp_model: ImbalancePriceModel,
        config: Optional[ImbalanceConfig] = None,
    ):
        self.model  = isp_model
        self.cfg    = config or ImbalanceConfig()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Run strategy
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Generate position signals from composite imbalance predictor.
        P&L = position * ISP change (simplified: position against ISP next period).
        """
        composite = self.model.composite_signal()
        isp       = self.model.isp
        nrv_sig   = self.model.nrv_signal()
        vre_sig   = self.model.vre_surprise_signal()

        roll_m    = composite.rolling(self.cfg.lookback).mean()
        roll_s    = composite.rolling(self.cfg.lookback).std().replace(0, np.nan)
        zscore    = (composite - roll_m) / roll_s

        position = pd.Series(0.0, index=isp.index)
        current  = 0

        for i in range(self.cfg.lookback, len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z): continue
            if current == 0:
                if z > self.cfg.entry_zscore:
                    current =  1    # bullish ISP expected → long
                elif z < -self.cfg.entry_zscore:
                    current = -1    # bearish ISP expected → short
            elif current == 1:
                if z < self.cfg.exit_zscore or z < -self.cfg.stop_zscore:
                    current = 0
            elif current == -1:
                if z > -self.cfg.exit_zscore or z > self.cfg.stop_zscore:
                    current = 0
            position.iloc[i] = current

        # P&L: position vs actual ISP change
        isp_change = isp.diff()
        daily_pnl  = position.shift(1) * isp_change * self.cfg.position_size_mw / 4  # 15-min period

        self.results = pd.DataFrame({
            "isp":          isp,
            "composite":    composite,
            "nrv_signal":   nrv_sig if nrv_sig is not None else np.nan,
            "vre_signal":   vre_sig if vre_sig is not None else np.nan,
            "zscore":       zscore,
            "position":     position,
            "isp_change":   isp_change,
            "daily_pnl":    daily_pnl,
            "cum_pnl":      daily_pnl.cumsum(),
        })
        return self.results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if self.results is None: raise RuntimeError("Call run() first.")
        pnl    = self.results["daily_pnl"].dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 96) if pnl.std() > 0 else 0.0
        dd     = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        isp_stats = self.model.isp_statistics()
        return {
            **isp_stats,
            "total_pnl_eur":    round(pnl.sum(), 2),
            "sharpe_ratio":     round(sharpe, 3),
            "max_drawdown":     round(dd, 2),
            "win_rate":         round((pnl > 0).mean(), 3),
            "n_long":           int((self.results["position"] == 1).sum()),
            "n_short":          int((self.results["position"] == -1).sum()),
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, figsize=(14, 13)) -> plt.Figure:
        if self.results is None: raise RuntimeError("Call run() first.")
        df  = self.results
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle("Imbalance Settlement Price Signal Strategy",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.fill_between(df.index, df["isp"], 0,
                        where=df["isp"] > 0, color="#c62828", alpha=0.5, label="ISP > 0")
        ax.fill_between(df.index, df["isp"], 0,
                        where=df["isp"] < 0, color="#1565c0", alpha=0.5, label="ISP < 0")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh")
        ax.set_title("Imbalance Settlement Price (ISP)", fontsize=10)

        ax = axes[1]
        if not df["nrv_signal"].isna().all():
            ax.plot(df.index, df["nrv_signal"],  color="#1565c0", lw=0.8, label="NRV signal", alpha=0.7)
        if not df["vre_signal"].isna().all():
            ax.plot(df.index, df["vre_signal"],  color="#2e7d32", lw=0.8, label="VRE signal", alpha=0.7)
        ax.plot(df.index, df["composite"], color="#e65100", lw=1.0, label="Composite")
        ax.axhline(0, color="black", lw=0.3)
        ax.legend(fontsize=8); ax.set_ylabel("Signal [-1,+1]")
        ax.set_title("Predictor Signals", fontsize=10)

        ax = axes[2]
        ax.plot(df.index, df["zscore"], color="#555", lw=0.8)
        ax.axhline(self.cfg.entry_zscore,  color="green", lw=0.8, ls="--")
        ax.axhline(-self.cfg.entry_zscore, color="red",   lw=0.8, ls="--")
        ax.axhline(0, color="black", lw=0.3)
        longs  = df[df["position"] == 1]
        shorts = df[df["position"] == -1]
        ax.scatter(longs.index,  longs["zscore"],  s=10, color="green", zorder=5)
        ax.scatter(shorts.index, shorts["zscore"], s=10, color="red",   zorder=5)
        ax.set_ylabel("Z-score"); ax.set_title("Signal Z-score & Trades", fontsize=10)

        ax = axes[3]
        ax.fill_between(df.index, df["cum_pnl"], 0,
                        where=df["cum_pnl"] >= 0, color="green", alpha=0.4)
        ax.fill_between(df.index, df["cum_pnl"], 0,
                        where=df["cum_pnl"] < 0,  color="red",   alpha=0.4)
        ax.plot(df.index, df["cum_pnl"], color="black", lw=0.8)
        ax.set_ylabel("€ cumul."); ax.set_title("Cumulative P&L", fontsize=10)
        ax.set_xlabel("Time")
        plt.tight_layout()
        return fig


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng    = np.random.default_rng(42)
    n      = 96 * 30    # 30 days × 96 quarter-hours
    times  = pd.date_range("2024-01-01", periods=n, freq="15min")

    # Simulate ISP: spiky mean-reverting process
    isp    = [60.0]
    for _ in range(n - 1):
        spike = rng.choice([0, 1], p=[0.95, 0.05])
        di    = 0.3 * (60 - isp[-1]) + rng.normal(0, 8) + spike * rng.choice([-1,1]) * rng.uniform(50,150)
        isp.append(np.clip(isp[-1] + di, -200, 500))
    isp_s  = pd.Series(isp, index=times, name="isp")

    # Predictors
    nrv      = pd.Series(rng.normal(0, 500, n), index=times)   # MW
    wind_s   = pd.Series(rng.normal(0, 300, n), index=times)   # MW surprise
    solar_s  = pd.Series(rng.normal(0, 150, n), index=times)
    load_s   = pd.Series(rng.normal(0, 200, n), index=times)
    id_px    = pd.Series(np.clip(60 + np.cumsum(rng.normal(0, 0.3, n)), 10, 300), index=times)

    model  = ImbalancePriceModel(
        imbalance_prices = isp_s,
        nrv              = nrv,
        wind_surprise    = wind_s,
        solar_surprise   = solar_s,
        load_surprise    = load_s,
        id_prices        = id_px,
        params           = BalancingMarketParams(country="DE"),
    )

    print("\n=== ISP Statistics ===")
    for k, v in model.isp_statistics().items():
        print(f"  {k:25s}: {v}")

    print("\n=== Signal-ISP Correlations ===")
    print(model.signal_isp_correlation().to_string())

    cfg    = ImbalanceConfig(lookback=20, entry_zscore=1.5, position_size_mw=10.0)
    strat  = ImbalanceSignalStrategy(isp_model=model, config=cfg)
    strat.run()
    stats  = strat.summary()

    print("\n=== Imbalance Signal Strategy — Summary ===")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    fig = strat.plot()
    fig.savefig("imbalance_signal.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → imbalance_signal.png")
