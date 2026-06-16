"""
Cross-Border Power Interconnector Arbitrage
============================================
Exploits price differentials between two power price zones
limited by the physical capacity of the interconnector between them.

Market rationale:
    In a perfectly integrated market, power prices across borders
    would converge. In reality, congestion on interconnectors
    prevents full arbitrage, creating persistent price spreads.

    Key European interconnectors:
        FR ↔ DE   ~3,000 MW NTC
        IT ↔ AT   ~   380 MW NTC (Brenner corridor)
        IT ↔ FR   ~  2,350 MW NTC
        UK ↔ FR   ~  2,000 MW (IFA1+IFA2)
        DE ↔ NL   ~  3,850 MW NTC
        NO ↔ DE   ~  1,400 MW (NordLink)

    NTC = Net Transfer Capacity (available for commercial flows)

Strategy logic:
    1. Compute spread: Price_A - Price_B
    2. If |spread| > transaction cost → arbitrage opportunity exists
    3. Size position as fraction of NTC (physical constraint)
    4. Mean-reversion signal: spread reverts when congestion eases
    5. Regime filter: distinguish congested vs uncongested hours

Physical constraint:
    Max position ≤ NTC × capacity_factor
    P&L = spread × volume_MW × hours

Data sources:
    - ENTSO-E Transparency Platform (free API, entsoe-py library)
    - EPEX SPOT day-ahead prices
    - GME (Italy)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class InterconnectorParams:
    """Physical and commercial parameters of an interconnector."""
    name: str               = "FR-DE"
    ntc_mw: float           = 3000.0        # Net Transfer Capacity [MW]
    capacity_factor: float  = 0.70          # fraction of NTC available for trading
    transaction_cost: float = 0.30          # €/MWh round-trip transaction cost
    zone_a: str             = "FR"
    zone_b: str             = "DE"

    @property
    def max_position_mw(self) -> float:
        return self.ntc_mw * self.capacity_factor


@dataclass
class InterconnectorConfig:
    """Strategy configuration."""
    lookback: int           = 24            # hours (for hourly data) or days
    entry_threshold: float  = 1.0          # €/MWh min spread to enter (above TC)
    entry_zscore: float     = 1.5
    exit_zscore: float      = 0.3
    stop_zscore: float      = 4.0
    use_congestion_filter: bool = True      # skip trades when congestion > threshold
    congestion_threshold: float = 0.90     # NTC utilisation above which we skip


class CrossBorderArbitrage:
    """
    Cross-border power interconnector arbitrage strategy.

    Parameters
    ----------
    price_a          : pd.Series   Power prices zone A [€/MWh].
    price_b          : pd.Series   Power prices zone B [€/MWh].
    ntc_utilisation  : pd.Series   Optional NTC utilisation ratio [0–1].
    params           : InterconnectorParams
    config           : InterconnectorConfig
    """

    def __init__(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        ntc_utilisation: Optional[pd.Series] = None,
        params: Optional[InterconnectorParams] = None,
        config: Optional[InterconnectorConfig] = None,
    ):
        self.price_a  = price_a.rename("price_A")
        self.price_b  = price_b.rename("price_B")
        self.ntc_util = ntc_utilisation
        self.params   = params or InterconnectorParams()
        self.cfg      = config or InterconnectorConfig()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Spread & opportunity detection
    # ------------------------------------------------------------------

    def compute_spread(self) -> pd.Series:
        """Price_A - Price_B [€/MWh]."""
        s = self.price_a - self.price_b
        s.name = "spread"
        return s

    def arbitrage_opportunity(self, spread: pd.Series) -> pd.Series:
        """
        Boolean series: True when |spread| > transaction cost
        i.e. a gross arbitrage opportunity exists.
        """
        opp = spread.abs() > self.params.transaction_cost
        opp.name = "arb_opportunity"
        return opp

    def net_spread(self, spread: pd.Series) -> pd.Series:
        """Spread net of transaction cost (signed)."""
        tc  = self.params.transaction_cost
        net = spread.copy()
        net[spread > 0]  -= tc
        net[spread < 0]  += tc
        net[spread.abs() <= tc] = 0.0
        net.name = "net_spread"
        return net

    # ------------------------------------------------------------------
    # Congestion analysis
    # ------------------------------------------------------------------

    def congestion_regime(self) -> Optional[pd.Series]:
        """
        Returns True on days/hours where NTC is heavily utilised
        (congestion prevents further arbitrage flows).
        """
        if self.ntc_util is None:
            return None
        congested = self.ntc_util >= self.cfg.congestion_threshold
        congested.name = "congested"
        return congested

    def congestion_statistics(self) -> dict:
        """Summary of congestion frequency."""
        if self.ntc_util is None:
            return {"ntc_data": "not provided"}
        return {
            "congestion_pct":   round((self.ntc_util >= self.cfg.congestion_threshold).mean() * 100, 1),
            "avg_utilisation":  round(self.ntc_util.mean() * 100, 1),
            "max_utilisation":  round(self.ntc_util.max() * 100, 1),
        }

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Generate trading signals.

        Direction:
            +1 = buy A, sell B  (spread expected to widen or revert from negative)
            -1 = sell A, buy B  (spread expected to narrow or revert from positive)

        Position sized as fraction of NTC — here simplified to ±1 unit.
        To scale to MW: multiply daily_pnl by params.max_position_mw.
        """
        spread      = self.compute_spread()
        net_sp      = self.net_spread(spread)
        roll_mean   = spread.rolling(self.cfg.lookback).mean()
        roll_std    = spread.rolling(self.cfg.lookback).std()
        zscore      = (spread - roll_mean) / roll_std.replace(0, np.nan)
        congested   = self.congestion_regime()
        arb_opp     = self.arbitrage_opportunity(spread)

        position = pd.Series(0.0, index=spread.index)
        current  = 0

        for i in range(self.cfg.lookback, len(zscore)):
            z   = zscore.iloc[i]
            ns  = net_sp.iloc[i]
            if np.isnan(z):
                continue

            # Congestion filter: skip entry if NTC saturated
            blocked = (self.cfg.use_congestion_filter and
                       congested is not None and congested.iloc[i])

            if current == 0 and not blocked:
                if z > self.cfg.entry_zscore and ns > self.cfg.entry_threshold:
                    current = -1    # sell A, buy B — expect spread to narrow
                elif z < -self.cfg.entry_zscore and (-ns) > self.cfg.entry_threshold:
                    current = 1     # buy A, sell B — expect spread to widen
            elif current == 1:
                if z >= -self.cfg.exit_zscore or z <= -self.cfg.stop_zscore:
                    current = 0
            elif current == -1:
                if z <= self.cfg.exit_zscore or z >= self.cfg.stop_zscore:
                    current = 0

            position.iloc[i] = current

        daily_pnl    = position.shift(1) * spread.diff()
        daily_pnl_mw = daily_pnl * self.params.max_position_mw   # scaled to MW

        self.results = pd.DataFrame({
            "price_A":      self.price_a,
            "price_B":      self.price_b,
            "spread":       spread,
            "net_spread":   net_sp,
            "ntc_util":     self.ntc_util if self.ntc_util is not None else np.nan,
            "arb_opp":      arb_opp,
            "roll_mean":    roll_mean,
            "zscore":       zscore,
            "position":     position,
            "daily_pnl":    daily_pnl,
            "daily_pnl_mw": daily_pnl_mw,
            "cum_pnl":      daily_pnl.cumsum(),
            "cum_pnl_mw":   daily_pnl_mw.cumsum(),
        })
        return self.results

    # ------------------------------------------------------------------
    # Opportunity analysis
    # ------------------------------------------------------------------

    def opportunity_analysis(self) -> pd.DataFrame:
        """
        Breakdown of arbitrage opportunity frequency and magnitude
        by calendar month.
        """
        spread = self.compute_spread()
        arb    = self.arbitrage_opportunity(spread)
        df = pd.DataFrame({
            "spread": spread,
            "arb":    arb,
            "month":  spread.index.month,
        })
        result = df.groupby("month").agg(
            arb_freq_pct  = ("arb",    lambda x: round(x.mean() * 100, 1)),
            avg_spread    = ("spread", lambda x: round(x.mean(), 2)),
            avg_abs_spread= ("spread", lambda x: round(x.abs().mean(), 2)),
            max_spread    = ("spread", lambda x: round(x.max(), 2)),
            min_spread    = ("spread", lambda x: round(x.min(), 2)),
        )
        result.index = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
        return result

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
        arb    = self.results["arb_opp"]
        return {
            "interconnector":       self.params.name,
            "ntc_mw":               self.params.ntc_mw,
            "max_position_mw":      self.params.max_position_mw,
            "spread_mean":          round(spread.mean(), 2),
            "spread_std":           round(spread.std(), 2),
            "arb_opportunity_pct":  round(arb.mean() * 100, 1),
            "total_pnl_unit":       round(pnl.sum(), 2),
            "total_pnl_eur":        round((self.results["daily_pnl_mw"]).sum(), 0),
            "sharpe_ratio":         round(sharpe, 3),
            "max_drawdown":         round(dd, 2),
            "win_rate":             round((pnl > 0).mean(), 3),
            **self.congestion_statistics(),
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, figsize=(14, 12)) -> plt.Figure:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        df  = self.results
        has_ntc = not df["ntc_util"].isna().all()
        nrows   = 5 if has_ntc else 4

        fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True)
        fig.suptitle(f"Cross-Border Arbitrage — {self.params.name} "
                     f"({self.params.zone_a} vs {self.params.zone_b})",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["price_A"], label=self.params.zone_a, color="#1565c0", lw=1.0)
        ax.plot(df.index, df["price_B"], label=self.params.zone_b, color="#c62828", lw=1.0)
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("Zone Prices", fontsize=10)

        ax = axes[1]
        ax.plot(df.index, df["spread"], color="#333", lw=0.9, label="Spread A-B")
        ax.axhline(self.params.transaction_cost,  color="orange", lw=0.8, ls="--", label=f"+TC ({self.params.transaction_cost}€)")
        ax.axhline(-self.params.transaction_cost, color="orange", lw=0.8, ls="--", label=f"-TC")
        ax.axhline(0, color="black", lw=0.4, ls=":")
        ax.fill_between(df.index, df["spread"],
                        self.params.transaction_cost,
                        where=df["spread"] > self.params.transaction_cost,
                        color="green", alpha=0.2, label="Arb opportunity (+)")
        ax.fill_between(df.index, df["spread"],
                        -self.params.transaction_cost,
                        where=df["spread"] < -self.params.transaction_cost,
                        color="red", alpha=0.2, label="Arb opportunity (-)")
        ax.legend(fontsize=7); ax.set_ylabel("€/MWh"); ax.set_title("Price Spread", fontsize=10)

        row = 2
        if has_ntc:
            ax = axes[row]
            ax.fill_between(df.index, df["ntc_util"] * 100, 0, color="#e65100", alpha=0.5)
            ax.axhline(self.cfg.congestion_threshold * 100, color="red", lw=0.8, ls="--",
                       label=f"Congestion threshold ({self.cfg.congestion_threshold*100:.0f}%)")
            ax.legend(fontsize=8); ax.set_ylabel("%")
            ax.set_title("NTC Utilisation", fontsize=10); row += 1

        ax = axes[row]
        ax.plot(df.index, df["zscore"], color="#555", lw=0.8)
        ax.axhline(self.cfg.entry_zscore,  color="red",   lw=0.8, ls="--")
        ax.axhline(-self.cfg.entry_zscore, color="green", lw=0.8, ls="--")
        ax.axhline(0, color="black", lw=0.3)
        longs  = df[df["position"] == 1]
        shorts = df[df["position"] == -1]
        ax.scatter(longs.index,  longs["zscore"],  s=14, color="green", zorder=5, label="Long A-B")
        ax.scatter(shorts.index, shorts["zscore"], s=14, color="red",   zorder=5, label="Short A-B")
        ax.legend(fontsize=8); ax.set_ylabel("Z-score"); ax.set_title("Signal", fontsize=10); row += 1

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
    rng   = np.random.default_rng(33)
    n     = 700
    dates = pd.date_range("2021-01-01", periods=n, freq="B")

    common  = np.cumsum(rng.normal(0, 1.5, n))
    price_fr = np.clip(80 + common + rng.normal(0, 4, n), 20, 400)
    price_de = np.clip(80 + common + rng.normal(0, 4, n) + 3, 20, 400)
    ntc_util = np.clip(0.6 + rng.normal(0, 0.15, n) +
                       0.2 * np.sin(2 * np.pi * np.arange(n) / 252), 0, 1)

    params = InterconnectorParams(name="FR-DE", ntc_mw=3000, zone_a="FR", zone_b="DE")
    cfg    = InterconnectorConfig(lookback=20, entry_zscore=1.5, use_congestion_filter=True)

    strat = CrossBorderArbitrage(
        price_a         = pd.Series(price_fr, index=dates),
        price_b         = pd.Series(price_de, index=dates),
        ntc_utilisation = pd.Series(ntc_util, index=dates),
        params=params, config=cfg,
    )

    results = strat.run()
    stats   = strat.summary()

    print("\n=== Cross-Border Arbitrage — Summary ===")
    for k, v in stats.items():
        print(f"  {k:28s}: {v}")

    print("\n  Monthly Opportunity Analysis:")
    print(strat.opportunity_analysis().to_string())

    fig = strat.plot()
    fig.savefig("cross_border_arb.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → cross_border_arb.png")
