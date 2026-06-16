"""
NBP vs TTF Spread — UK vs Continental Gas
==========================================
Trades the spread between the UK National Balancing Point (NBP)
and the Dutch Title Transfer Facility (TTF), the two most historically
significant European gas hubs.

Market drivers:
    The NBP-TTF spread is driven by:

    1. LNG flows:
       UK has significant LNG import capacity (South Hook, Dragon, Isle of Grain).
       High global LNG supply → NBP bearish vs TTF.
       Tight LNG → NBP premium over TTF.

    2. Interconnector (IUK) capacity:
       The Interconnector UK pipeline links Bacton (UK) to Zeebrugge (Belgium/TTF).
       When flowing UK→Continent: NBP premium, TTF cheaper.
       When flowing Continent→UK: TTF premium, NBP cheaper.
       Capacity ~20 bcm/year bidirectional.

    3. UK storage:
       Post-Rough closure (2017), UK has very limited storage.
       UK more reliant on LNG and IUK flows → more volatile NBP.

    4. Seasonal demand:
       UK winter demand, Norwegian pipeline supply to continent.

    5. Brexit effect:
       Post-2021: some liquidity migration from NBP to TTF.

Unit conversion:
    NBP: pence per therm (p/therm)
    TTF: €/MWh
    Conversion: 1 therm = 29.307 kWh; apply GBP/EUR FX

Strategy:
    Statistical mean-reversion with LNG, IUK flow, and storage overlays.
    Regime filter: distinguish LNG-flush vs LNG-tight market regimes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class NBPTTFConfig:
    """Configuration for NBP-TTF spread strategy."""
    lookback: int           = 20
    entry_zscore: float     = 1.8
    exit_zscore: float      = 0.4
    stop_zscore: float      = 3.5
    iuk_capacity_gwh: float = 515.0     # IUK daily capacity [GWh/day]
    lng_high_threshold: float = 0.70    # fraction of UK regasification capacity


class NBPTTFSpread:
    """
    NBP vs TTF spread strategy (both in €/MWh after conversion).

    Parameters
    ----------
    nbp_eur_mwh   : pd.Series   NBP price converted to €/MWh.
    ttf_eur_mwh   : pd.Series   TTF price [€/MWh].
    iuk_flow      : pd.Series   IUK daily flow [GWh/day], positive = UK→Continent.
    lng_sendout   : pd.Series   UK LNG send-out [GWh/day].
    uk_storage_pct: pd.Series   UK storage fill [%].
    config        : NBPTTFConfig
    """

    def __init__(
        self,
        nbp_eur_mwh: pd.Series,
        ttf_eur_mwh: pd.Series,
        iuk_flow: Optional[pd.Series] = None,
        lng_sendout: Optional[pd.Series] = None,
        uk_storage_pct: Optional[pd.Series] = None,
        config: Optional[NBPTTFConfig] = None,
    ):
        self.nbp     = nbp_eur_mwh.rename("NBP")
        self.ttf     = ttf_eur_mwh.rename("TTF")
        self.iuk     = iuk_flow
        self.lng     = lng_sendout
        self.uk_stor = uk_storage_pct
        self.cfg     = config or NBPTTFConfig()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Spread
    # ------------------------------------------------------------------

    def compute_spread(self) -> pd.Series:
        """NBP - TTF [€/MWh]. Positive = NBP premium."""
        s = self.nbp - self.ttf
        s.name = "nbp_ttf_spread"
        return s

    # ------------------------------------------------------------------
    # Fundamental signals
    # ------------------------------------------------------------------

    def iuk_signal(self) -> Optional[pd.Series]:
        """
        IUK flow direction signal.
        UK→Continent flow: NBP at premium → lean short spread.
        Continent→UK flow: NBP at discount → lean long spread.
        """
        if self.iuk is None:
            return None
        sig = pd.Series(0, index=self.iuk.index, name="iuk_signal")
        sig[self.iuk > self.cfg.iuk_capacity_gwh * 0.5]  = -1   # UK exports → NBP expensive
        sig[self.iuk < -self.cfg.iuk_capacity_gwh * 0.5] =  1   # Cont exports → NBP cheap
        return sig

    def lng_regime(self) -> Optional[pd.Series]:
        """
        LNG regime classification.
        High UK LNG send-out → UK oversupply → NBP bearish → short spread.
        """
        if self.lng is None:
            return None
        max_cap = 1500   # GWh/day approximate UK max regasification
        regime  = pd.Series("normal", index=self.lng.index, name="lng_regime")
        regime[self.lng > max_cap * self.cfg.lng_high_threshold] = "LNG_flush"
        regime[self.lng < max_cap * 0.10]                        = "LNG_tight"
        return regime

    def storage_signal(self) -> Optional[pd.Series]:
        """
        UK storage signal. Low storage → NBP bullish → long spread.
        """
        if self.uk_stor is None:
            return None
        sig = pd.Series(0, index=self.uk_stor.index, name="uk_stor_signal")
        sig[self.uk_stor < 25] =  1    # low UK storage → bullish NBP
        sig[self.uk_stor > 80] = -1    # high UK storage → bearish NBP
        return sig

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        spread    = self.compute_spread()
        roll_mean = spread.rolling(self.cfg.lookback).mean()
        roll_std  = spread.rolling(self.cfg.lookback).std()
        zscore    = (spread - roll_mean) / roll_std.replace(0, np.nan)

        iuk_sig   = self.iuk_signal()
        lng_reg   = self.lng_regime()
        stor_sig  = self.storage_signal()

        position = pd.Series(0.0, index=spread.index)
        current  = 0

        for i in range(self.cfg.lookback, len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z):
                continue

            # Aggregate fundamental bias
            bias = 0
            if iuk_sig  is not None: bias += iuk_sig.iloc[i]
            if stor_sig is not None: bias += stor_sig.iloc[i]
            # LNG regime: flush = bearish NBP = bias -1
            if lng_reg is not None:
                if lng_reg.iloc[i] == "LNG_flush": bias -= 1
                elif lng_reg.iloc[i] == "LNG_tight": bias += 1
            bias = np.clip(bias, -2, 2)

            if current == 0:
                if z > self.cfg.entry_zscore and bias <= 0:
                    current = -1
                elif z < -self.cfg.entry_zscore and bias >= 0:
                    current = 1
            elif current == 1:
                if z >= -self.cfg.exit_zscore or z <= -self.cfg.stop_zscore:
                    current = 0
            elif current == -1:
                if z <= self.cfg.exit_zscore or z >= self.cfg.stop_zscore:
                    current = 0

            position.iloc[i] = current

        daily_pnl = position.shift(1) * spread.diff()

        self.results = pd.DataFrame({
            "NBP":          self.nbp,
            "TTF":          self.ttf,
            "spread":       spread,
            "iuk_flow":     self.iuk     if self.iuk     is not None else np.nan,
            "lng_sendout":  self.lng     if self.lng     is not None else np.nan,
            "uk_storage":   self.uk_stor if self.uk_stor is not None else np.nan,
            "roll_mean":    roll_mean,
            "zscore":       zscore,
            "position":     position,
            "daily_pnl":    daily_pnl,
            "cum_pnl":      daily_pnl.cumsum(),
        })
        return self.results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        pnl    = self.results["daily_pnl"].dropna()
        spread = self.results["spread"].dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        dd     = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        return {
            "spread_pair":      "NBP - TTF",
            "nbp_premium_pct":  round((spread > 0).mean() * 100, 1),
            "ttf_premium_pct":  round((spread < 0).mean() * 100, 1),
            "spread_mean":      round(spread.mean(), 3),
            "spread_std":       round(spread.std(), 3),
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
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle("NBP vs TTF Spread — UK vs Continental Gas",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["NBP"], label="NBP (€/MWh)", color="#6a1b9a", lw=1.0)
        ax.plot(df.index, df["TTF"], label="TTF (€/MWh)", color="#1565c0", lw=1.0)
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("Hub Prices", fontsize=10)

        ax = axes[1]
        ax.plot(df.index, df["spread"],    color="#333",   lw=0.9, label="NBP-TTF Spread")
        ax.plot(df.index, df["roll_mean"], color="#9c27b0", lw=0.8, ls="--", label="Rolling Mean")
        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.fill_between(df.index, df["spread"], 0,
                        where=df["spread"] > 0, color="#6a1b9a", alpha=0.18, label="NBP premium")
        ax.fill_between(df.index, df["spread"], 0,
                        where=df["spread"] < 0, color="#1565c0", alpha=0.18, label="TTF premium")
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("NBP-TTF Spread", fontsize=10)

        ax = axes[2]
        ax.plot(df.index, df["zscore"], color="#555", lw=0.8)
        ax.axhline(self.cfg.entry_zscore,  color="red",   lw=0.8, ls="--")
        ax.axhline(-self.cfg.entry_zscore, color="green", lw=0.8, ls="--")
        ax.axhline(0, color="black", lw=0.3)
        longs  = df[df["position"] == 1]
        shorts = df[df["position"] == -1]
        ax.scatter(longs.index,  longs["zscore"],  s=14, color="green", zorder=5, label="Long NBP")
        ax.scatter(shorts.index, shorts["zscore"], s=14, color="red",   zorder=5, label="Short NBP")
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
    rng   = np.random.default_rng(88)
    n     = 700
    dates = pd.date_range("2021-01-01", periods=n, freq="B")

    common  = np.cumsum(rng.normal(0, 1.8, n))
    ttf     = np.clip(55 + common + rng.normal(0, 0.8, n), 15, 350)
    nbp     = np.clip(55 + common + rng.normal(0, 1.5, n) - 1.0, 15, 350)

    iuk     = rng.normal(0, 200, n)          # GWh/day, pos=UK→Continent
    lng     = np.clip(300 + rng.normal(0, 150, n), 0, 1200)
    uk_stor = np.clip(50 + 30*np.sin(np.linspace(0, 4*np.pi, n)) + rng.normal(0, 5, n), 5, 100)

    cfg = NBPTTFConfig(lookback=20, entry_zscore=1.8)
    strat = NBPTTFSpread(
        nbp_eur_mwh    = pd.Series(nbp,     index=dates),
        ttf_eur_mwh    = pd.Series(ttf,     index=dates),
        iuk_flow       = pd.Series(iuk,     index=dates),
        lng_sendout    = pd.Series(lng,     index=dates),
        uk_storage_pct = pd.Series(uk_stor, index=dates),
        config=cfg,
    )

    results = strat.run()
    stats   = strat.summary()
    print("\n=== NBP-TTF Spread — Summary ===")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    fig = strat.plot()
    fig.savefig("nbp_ttf_spread.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → nbp_ttf_spread.png")
