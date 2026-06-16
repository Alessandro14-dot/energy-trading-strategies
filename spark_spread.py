"""
Spark Spread & Clean Spark Spread Strategy
==========================================
Computes and backtests the spark spread between natural gas and power prices.
The clean spark spread adjusts for carbon (EUA) cost.

Formula:
    Spark Spread     = Power Price - (Gas Price / Efficiency)
    Clean Spark Spread = Spark Spread - (EUA Price * CO2_Factor)

Typical values (Europe, CCGT):
    Efficiency  = 0.50 (50%)
    CO2_Factor  = 0.37 t CO2 / MWh_th  → 0.37 / 0.50 = 0.74 t CO2 / MWh_el
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional


@dataclass
class CCGTParams:
    """Parameters for a Combined Cycle Gas Turbine unit."""
    efficiency: float = 0.50          # thermal efficiency (LHV)
    co2_factor_th: float = 0.37       # t CO2 / MWh_thermal (gas)
    min_run_hours: int = 4            # minimum run block (optional)
    variable_om: float = 2.0         # €/MWh variable O&M


class SparkSpreadStrategy:
    """
    Compute spark spread and clean spark spread time series.
    Generates trading signals based on spread thresholds.

    Parameters
    ----------
    power_prices : pd.Series
        Day-ahead or forward power prices [€/MWh].
    gas_prices : pd.Series
        Gas prices at relevant hub (TTF, NBP, NCG) [€/MWh].
    eua_prices : pd.Series
        EU Allowances (EUA) prices [€/t CO2]. Optional for clean spread.
    params : CCGTParams
        Plant technical parameters.
    """

    def __init__(
        self,
        power_prices: pd.Series,
        gas_prices: pd.Series,
        eua_prices: Optional[pd.Series] = None,
        params: Optional[CCGTParams] = None,
    ):
        self.power = power_prices
        self.gas = gas_prices
        self.eua = eua_prices
        self.params = params or CCGTParams()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Spread computation
    # ------------------------------------------------------------------

    def compute_spark_spread(self) -> pd.Series:
        """Raw spark spread [€/MWh of power]."""
        fuel_cost = self.gas / self.params.efficiency
        spread = self.power - fuel_cost - self.params.variable_om
        spread.name = "spark_spread"
        return spread

    def compute_clean_spark_spread(self) -> pd.Series:
        """
        Clean spark spread — spark spread minus carbon cost.
        Requires EUA prices to be provided.
        """
        if self.eua is None:
            raise ValueError("EUA prices required to compute clean spark spread.")
        co2_el = self.params.co2_factor_th / self.params.efficiency  # t CO2 / MWh_el
        carbon_cost = self.eua * co2_el
        css = self.compute_spark_spread() - carbon_cost
        css.name = "clean_spark_spread"
        return css

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        long_threshold: float = 3.0,
        short_threshold: float = -2.0,
        lookback: int = 20,
    ) -> pd.DataFrame:
        """
        Generate long/short signals based on rolling z-score of the spread.

        - Long (run plant / buy power + sell gas):  z-score > long_threshold
        - Short (don't run / sell forward):         z-score < short_threshold
        - Flat:                                     between thresholds

        Parameters
        ----------
        long_threshold : float
            Z-score above which we go long the spread.
        short_threshold : float
            Z-score below which we go short the spread.
        lookback : int
            Rolling window (days) for z-score calculation.
        """
        spread = self.compute_spark_spread()

        try:
            css = self.compute_clean_spark_spread()
        except ValueError:
            css = pd.Series(np.nan, index=spread.index, name="clean_spark_spread")

        roll_mean = spread.rolling(lookback).mean()
        roll_std = spread.rolling(lookback).std()
        zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)

        signal = pd.Series(0, index=spread.index, name="signal")
        signal[zscore > long_threshold] = 1
        signal[zscore < short_threshold] = -1

        pnl = signal.shift(1) * spread.diff()

        self.results = pd.DataFrame(
            {
                "power": self.power,
                "gas": self.gas,
                "eua": self.eua if self.eua is not None else np.nan,
                "spark_spread": spread,
                "clean_spark_spread": css,
                "zscore": zscore,
                "signal": signal,
                "daily_pnl": pnl,
                "cumulative_pnl": pnl.cumsum(),
            }
        )
        return self.results

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def summary_statistics(self) -> dict:
        """Return key backtest statistics."""
        if self.results is None:
            raise RuntimeError("Run generate_signals() first.")

        pnl = self.results["daily_pnl"].dropna()
        spread = self.results["spark_spread"].dropna()

        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        max_dd = (pnl.cumsum() - pnl.cumsum().cummax()).min()

        return {
            "spread_mean": round(spread.mean(), 2),
            "spread_std": round(spread.std(), 2),
            "spread_min": round(spread.min(), 2),
            "spread_max": round(spread.max(), 2),
            "total_pnl": round(pnl.sum(), 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(max_dd, 2),
            "win_rate": round((pnl > 0).mean(), 3),
            "n_signals_long": int((self.results["signal"] == 1).sum()),
            "n_signals_short": int((self.results["signal"] == -1).sum()),
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, figsize: tuple = (14, 10)) -> plt.Figure:
        """Plot spread, z-score, signals, and cumulative P&L."""
        if self.results is None:
            raise RuntimeError("Run generate_signals() first.")

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle("Spark Spread Strategy", fontsize=14, fontweight="bold")

        df = self.results

        # Panel 1: Prices
        ax = axes[0]
        ax.plot(df.index, df["power"], label="Power (€/MWh)", color="#1a73e8")
        ax2 = ax.twinx()
        ax2.plot(df.index, df["gas"], label="Gas TTF (€/MWh)", color="#e67700", alpha=0.7)
        ax.set_ylabel("Power €/MWh")
        ax2.set_ylabel("Gas €/MWh")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.set_title("Input Prices", fontsize=10)

        # Panel 2: Spark Spread
        ax = axes[1]
        ax.plot(df.index, df["spark_spread"], color="#333", linewidth=0.8, label="Spark Spread")
        if not df["clean_spark_spread"].isna().all():
            ax.plot(df.index, df["clean_spark_spread"], color="#e74c3c", linewidth=0.8, label="Clean Spark Spread")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=8)
        ax.set_ylabel("€/MWh")
        ax.set_title("Spark Spread", fontsize=10)

        # Panel 3: Z-score & Signals
        ax = axes[2]
        ax.plot(df.index, df["zscore"], color="#555", linewidth=0.8)
        ax.axhline(3.0, color="green", linewidth=0.8, linestyle="--", alpha=0.8)
        ax.axhline(-2.0, color="red", linewidth=0.8, linestyle="--", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.3)
        longs = df[df["signal"] == 1]
        shorts = df[df["signal"] == -1]
        ax.scatter(longs.index, longs["zscore"], color="green", s=15, zorder=5, label="Long")
        ax.scatter(shorts.index, shorts["zscore"], color="red", s=15, zorder=5, label="Short")
        ax.legend(fontsize=8)
        ax.set_ylabel("Z-score")
        ax.set_title("Signal Z-score", fontsize=10)

        # Panel 4: Cumulative P&L
        ax = axes[3]
        ax.fill_between(df.index, df["cumulative_pnl"], 0,
                        where=df["cumulative_pnl"] >= 0, alpha=0.4, color="green")
        ax.fill_between(df.index, df["cumulative_pnl"], 0,
                        where=df["cumulative_pnl"] < 0, alpha=0.4, color="red")
        ax.plot(df.index, df["cumulative_pnl"], color="black", linewidth=0.8)
        ax.set_ylabel("€/MWh cumulative")
        ax.set_title("Cumulative P&L", fontsize=10)
        ax.set_xlabel("Date")

        plt.tight_layout()
        return fig


# ------------------------------------------------------------------
# Quick demo with synthetic data
# ------------------------------------------------------------------

def generate_sample_data(n_days: int = 500, seed: int = 42) -> tuple:
    """Generate synthetic correlated power, gas, EUA price series."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")

    gas = 30 + np.cumsum(rng.normal(0, 1.5, n_days))
    gas = np.clip(gas, 10, 300)

    power_noise = rng.normal(0, 3, n_days)
    power = gas / 0.50 * 1.02 + power_noise + 5  # correlated to gas
    power = np.clip(power, 20, 600)

    eua = 60 + np.cumsum(rng.normal(0, 0.8, n_days))
    eua = np.clip(eua, 20, 130)

    return (
        pd.Series(power, index=dates, name="power"),
        pd.Series(gas, index=dates, name="gas"),
        pd.Series(eua, index=dates, name="eua"),
    )


if __name__ == "__main__":
    power, gas, eua = generate_sample_data(n_days=750)

    strategy = SparkSpreadStrategy(
        power_prices=power,
        gas_prices=gas,
        eua_prices=eua,
        params=CCGTParams(efficiency=0.50, co2_factor_th=0.37),
    )

    results = strategy.generate_signals(long_threshold=1.5, short_threshold=-1.0, lookback=30)
    stats = strategy.summary_statistics()

    print("\n=== Spark Spread Strategy — Summary ===")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    fig = strategy.plot()
    fig.savefig("spark_spread_backtest.png", dpi=150, bbox_inches="tight")
    print("\nChart saved to spark_spread_backtest.png")
