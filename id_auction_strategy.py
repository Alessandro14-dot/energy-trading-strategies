"""
Intraday Auction Strategy — EPEX SPOT & GME
=============================================
Systematic trading strategy for European power intraday auctions.

Market structure:
    Day-ahead auction results are published around 12:30 CET (D-1).
    After DA auction, intraday markets open for continuous and auction trading.

    Key intraday auction sessions (European):

    EPEX SPOT (DE, FR, NL, BE, AT, CH):
        - ID1 Auction:  15:00 CET (D-1) → delivery D, all hours
        - ID2 Auction:  22:00 CET (D-1) → delivery D, all hours
        - ID3 Auction:  10:00 CET (D)   → delivery D, remaining hours
        Continuous ID: opens 15:00 CET (D-1), gate closure H-5 or H-1

    GME (Italy — Mercato Infragiornaliero, MI):
        - MI1: 12:55 CET (D-1) → D all hours
        - MI2: 15:25 CET (D-1) → D all hours
        - MI3: 17:25 CET (D-1) → D all hours
        - MI4: 05:25 CET (D)   → D remaining hours (10-24)
        - MI5: 08:25 CET (D)   → D remaining hours (13-24)
        - MI6: 11:25 CET (D)   → D remaining hours (16-24)
        - MI7: 14:25 CET (D)   → D remaining hours (19-24)

Strategy logic:
    1. PRICE REVERSION:
       Intraday prices tend to revert toward DA price.
       Large DA-ID deviations are mean-reverting opportunities.

    2. FORECASTING DEVIATION TRADING:
       If new renewable forecast deviates significantly from DA forecast,
       intraday price will adjust. Trade in direction of deviation.

    3. AUCTION-TO-AUCTION SPREAD:
       Price differences between successive auction sessions (ID1 vs ID2)
       reflect new information. Trade the spread.

    4. PRICE ZONE CONVERGENCE:
       Italian zonal prices (NORD, SUD, CSUD) must converge to national
       reference price (PUN) absent congestion. Trade convergence.

Key signals:
    - DA vs ID spread z-score
    - Renewable generation surprise (forecast vs actual)
    - Residual demand at each auction time
    - Congestion indicator (inter-zonal price differences)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal


@dataclass
class AuctionSession:
    """Single intraday auction session specification."""
    name: str
    market: str              # "EPEX" or "GME"
    time_cet: str            # auction time e.g. "15:00"
    day_offset: int          # 0 = day D, -1 = day D-1
    delivery_hours: tuple    # (start_hour, end_hour) or None for all
    gate_closure_min: int    # minutes before delivery


# Default auction calendars
EPEX_SESSIONS = [
    AuctionSession("ID1", "EPEX", "15:00", -1, (0, 23), 60),
    AuctionSession("ID2", "EPEX", "22:00", -1, (0, 23), 60),
    AuctionSession("ID3", "EPEX", "10:00",  0, (0, 23), 60),
]

GME_SESSIONS = [
    AuctionSession("MI1", "GME", "12:55", -1, (0, 23), 55),
    AuctionSession("MI2", "GME", "15:25", -1, (0, 23), 55),
    AuctionSession("MI3", "GME", "17:25", -1, (0, 23), 55),
    AuctionSession("MI4", "GME", "05:25",  0, (9, 23), 55),
    AuctionSession("MI5", "GME", "08:25",  0, (12, 23), 55),
    AuctionSession("MI6", "GME", "11:25",  0, (15, 23), 55),
    AuctionSession("MI7", "GME", "14:25",  0, (18, 23), 55),
]


@dataclass
class IDConfig:
    """Intraday auction strategy configuration."""
    market: Literal["EPEX", "GME"] = "EPEX"
    zone: str                       = "DE_LU"
    lookback: int                   = 20        # days for rolling stats
    entry_zscore: float             = 1.5
    exit_zscore: float              = 0.3
    stop_zscore: float              = 3.5
    max_position_mwh: float         = 10.0      # max MW per auction
    transaction_cost: float         = 0.50      # €/MWh round-trip


class IDPriceReversionStrategy:
    """
    Intraday auction price reversion strategy.

    Core idea: Large deviations between DA and ID auction prices tend to
    mean-revert across successive ID sessions as new information is absorbed.

    Signals:
        1. DA vs ID1 spread → trade ID2
        2. ID1 vs ID2 spread → trade ID3
        3. Renewable surprise signal

    Parameters
    ----------
    da_prices       : pd.DataFrame  DA hourly prices [€/MWh], index=date, cols=hours
    id1_prices      : pd.DataFrame  ID1 auction prices
    id2_prices      : pd.DataFrame  ID2 auction prices
    wind_surprise   : pd.Series     Wind forecast surprise [MW] (actual - forecast)
    solar_surprise  : pd.Series     Solar forecast surprise [MW]
    config          : IDConfig
    """

    def __init__(
        self,
        da_prices: pd.DataFrame,
        id1_prices: pd.DataFrame,
        id2_prices: Optional[pd.DataFrame] = None,
        wind_surprise: Optional[pd.Series] = None,
        solar_surprise: Optional[pd.Series] = None,
        config: Optional[IDConfig] = None,
    ):
        self.da   = da_prices
        self.id1  = id1_prices
        self.id2  = id2_prices
        self.wind_surp  = wind_surprise
        self.solar_surp = solar_surprise
        self.cfg  = config or IDConfig()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Spread computation
    # ------------------------------------------------------------------

    def da_id1_spread(self) -> pd.Series:
        """
        Daily average DA - ID1 spread [€/MWh].
        Positive: DA prices above ID1 (bullish signal for ID2).
        """
        common = self.da.index.intersection(self.id1.index)
        spread = (self.da.loc[common].mean(axis=1) -
                  self.id1.loc[common].mean(axis=1))
        spread.name = "da_id1_spread"
        return spread

    def id1_id2_spread(self) -> Optional[pd.Series]:
        """ID1 - ID2 spread (if ID2 available)."""
        if self.id2 is None:
            return None
        common = self.id1.index.intersection(self.id2.index)
        spread = (self.id1.loc[common].mean(axis=1) -
                  self.id2.loc[common].mean(axis=1))
        spread.name = "id1_id2_spread"
        return spread

    # ------------------------------------------------------------------
    # Renewable surprise signal
    # ------------------------------------------------------------------

    def vre_surprise_signal(self) -> Optional[pd.Series]:
        """
        Composite renewable surprise signal.
        Positive surprise (more VRE than expected) → bearish power → short signal.
        Negative surprise → bullish power → long signal.
        Returns normalised signal [-1, +1].
        """
        if self.wind_surp is None and self.solar_surp is None:
            return None
        vre = pd.Series(0.0, index=self.da.index)
        if self.wind_surp is not None:
            vre = vre.add(self.wind_surp.reindex(vre.index).fillna(0), fill_value=0)
        if self.solar_surp is not None:
            vre = vre.add(self.solar_surp.reindex(vre.index).fillna(0), fill_value=0)
        # Normalise by rolling std
        vre_z = vre / vre.rolling(self.cfg.lookback).std().replace(0, np.nan)
        vre_z = vre_z.clip(-3, 3) / 3   # → [-1, +1]
        vre_z.name = "vre_surprise"
        return vre_z

    # ------------------------------------------------------------------
    # Hourly price profile analysis
    # ------------------------------------------------------------------

    def peak_offpeak_spread(self, prices: pd.DataFrame,
                             peak_hours=(8, 20)) -> pd.Series:
        """
        Daily peak (h8-h20) vs off-peak spread [€/MWh].
        Structural signal for intraday shape trading.
        """
        peak_cols   = [h for h in prices.columns if peak_hours[0] <= h < peak_hours[1]]
        offpk_cols  = [h for h in prices.columns if h not in peak_cols]
        if not peak_cols or not offpk_cols:
            return pd.Series(np.nan, index=prices.index)
        spread = prices[peak_cols].mean(axis=1) - prices[offpk_cols].mean(axis=1)
        spread.name = "peak_offpeak_spread"
        return spread

    def hourly_pattern(self, prices: pd.DataFrame) -> pd.Series:
        """Average hourly price pattern across all days."""
        return prices.mean(axis=0).rename("avg_price_by_hour")

    # ------------------------------------------------------------------
    # Run strategy
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Generate daily P&L from DA-ID1 spread mean reversion.
        Position: long ID1 / short DA  or  short ID1 / long DA.
        """
        spread   = self.da_id1_spread()
        roll_m   = spread.rolling(self.cfg.lookback).mean()
        roll_s   = spread.rolling(self.cfg.lookback).std()
        zscore   = (spread - roll_m) / roll_s.replace(0, np.nan)
        vre_sig  = self.vre_surprise_signal()
        id12_s   = self.id1_id2_spread()

        position = pd.Series(0.0, index=spread.index)
        current  = 0

        for i in range(self.cfg.lookback, len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z): continue

            # VRE overlay bias
            bias = 0
            if vre_sig is not None and i < len(vre_sig):
                bias = -vre_sig.iloc[i]   # more VRE → short bias

            if current == 0:
                if z > self.cfg.entry_zscore and bias <= 0:
                    current = -1    # DA >> ID1: buy ID2, expect reversion
                elif z < -self.cfg.entry_zscore and bias >= 0:
                    current =  1    # DA << ID1: sell ID2, expect reversion
            elif current == 1:
                if z >= -self.cfg.exit_zscore or z <= -self.cfg.stop_zscore:
                    current = 0
            elif current == -1:
                if z <=  self.cfg.exit_zscore or z >=  self.cfg.stop_zscore:
                    current = 0
            position.iloc[i] = current

        # P&L: position * next day's spread change (simplified)
        daily_pnl = position.shift(1) * spread.diff()
        # Apply transaction cost
        trades    = position.diff().abs()
        daily_pnl -= trades * self.cfg.transaction_cost / 2

        self.results = pd.DataFrame({
            "da_price":      self.da.mean(axis=1),
            "id1_price":     self.id1.mean(axis=1),
            "da_id1_spread": spread,
            "id12_spread":   id12_s if id12_s is not None else np.nan,
            "vre_surprise":  vre_sig if vre_sig is not None else np.nan,
            "roll_mean":     roll_m,
            "zscore":        zscore,
            "position":      position,
            "daily_pnl":     daily_pnl,
            "cum_pnl":       daily_pnl.cumsum(),
        })
        return self.results

    # ------------------------------------------------------------------
    # Auction performance analytics
    # ------------------------------------------------------------------

    def auction_statistics(self) -> pd.DataFrame:
        """
        Summary statistics of DA vs ID price differences by hour.
        Identifies which hours show the largest systematic deviations.
        """
        common = self.da.index.intersection(self.id1.index)
        diff   = self.da.loc[common] - self.id1.loc[common]
        stats  = pd.DataFrame({
            "mean_spread":    diff.mean(),
            "std_spread":     diff.std(),
            "pct_positive":   (diff > 0).mean() * 100,
            "max_spread":     diff.max(),
            "min_spread":     diff.min(),
        })
        stats.index.name = "hour"
        return stats.round(3)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if self.results is None: raise RuntimeError("Call run() first.")
        pnl    = self.results["daily_pnl"].dropna()
        spread = self.results["da_id1_spread"].dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        dd     = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        return {
            "market":               self.cfg.market,
            "zone":                 self.cfg.zone,
            "avg_da_id1_spread":    round(spread.mean(), 3),
            "std_da_id1_spread":    round(spread.std(), 3),
            "da_above_id_pct":      round((spread > 0).mean() * 100, 1),
            "total_pnl":            round(pnl.sum(), 3),
            "sharpe_ratio":         round(sharpe, 3),
            "max_drawdown":         round(dd, 3),
            "win_rate":             round((pnl > 0).mean(), 3),
            "n_long":               int((self.results["position"] == 1).sum()),
            "n_short":              int((self.results["position"] == -1).sum()),
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, figsize=(14, 13)) -> plt.Figure:
        if self.results is None: raise RuntimeError("Call run() first.")
        df  = self.results
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle(f"Intraday Auction Strategy — {self.cfg.market} {self.cfg.zone}",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["da_price"],  label="DA Price",  color="#1565c0", lw=1.0)
        ax.plot(df.index, df["id1_price"], label="ID1 Price", color="#e65100", lw=1.0, ls="--")
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("DA vs ID1 Prices", fontsize=10)

        ax = axes[1]
        ax.plot(df.index, df["da_id1_spread"], color="#333", lw=0.9, label="DA-ID1 Spread")
        ax.plot(df.index, df["roll_mean"],      color="#9c27b0", lw=0.8, ls="--", label="Rolling Mean")
        ax.axhline(0, color="black", lw=0.5, ls=":")
        ax.fill_between(df.index, df["da_id1_spread"], 0,
                        where=df["da_id1_spread"] > 0, color="#1565c0", alpha=0.18)
        ax.fill_between(df.index, df["da_id1_spread"], 0,
                        where=df["da_id1_spread"] < 0, color="#e65100", alpha=0.18)
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh"); ax.set_title("DA-ID1 Spread", fontsize=10)

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
    rng   = np.random.default_rng(42)
    n     = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    hours = list(range(24))

    # Synthetic DA and ID1 hourly price matrices
    base_price = 80 + np.cumsum(rng.normal(0, 1.5, n))
    base_price = np.clip(base_price, 20, 300)

    # DA: base + hourly shape
    shape = np.array([0.7,0.6,0.55,0.55,0.6,0.75,0.95,1.1,1.15,1.1,1.05,1.0,
                      0.95,0.95,1.0,1.05,1.1,1.15,1.1,1.0,0.9,0.8,0.75,0.7])
    da_matrix  = np.outer(base_price, shape) + rng.normal(0, 3, (n, 24))
    id1_matrix = da_matrix + rng.normal(0, 4, (n, 24))   # ID1 deviates from DA

    da_df  = pd.DataFrame(da_matrix,  index=dates, columns=hours)
    id1_df = pd.DataFrame(id1_matrix, index=dates, columns=hours)

    # Renewable surprises
    wind_surp  = pd.Series(rng.normal(0, 500, n), index=dates)
    solar_surp = pd.Series(rng.normal(0, 300, n), index=dates)

    cfg   = IDConfig(market="EPEX", zone="DE_LU", lookback=20, entry_zscore=1.5)
    strat = IDPriceReversionStrategy(
        da_prices     = da_df,
        id1_prices    = id1_df,
        wind_surprise = wind_surp,
        solar_surprise= solar_surp,
        config=cfg,
    )

    results = strat.run()
    stats   = strat.summary()
    print("\n=== Intraday Auction Strategy — Summary ===")
    for k, v in stats.items():
        print(f"  {k:28s}: {v}")

    print("\n  Hourly auction statistics (first 6 hours):")
    print(strat.auction_statistics().head(6).to_string())

    fig = strat.plot()
    fig.savefig("id_auction_strategy.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → id_auction_strategy.png")
