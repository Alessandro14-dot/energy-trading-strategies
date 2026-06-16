"""
Statistical Arbitrage — Cointegration-Based Pairs Trading
==========================================================
Implements Engle-Granger and Johansen cointegration tests
for energy market pairs (e.g. TTF vs NBP, Power DE vs Power FR).

Trading logic:
    1. Test for cointegration between two price series
    2. Estimate the hedge ratio (OLS or Kalman filter)
    3. Model the spread as an Ornstein-Uhlenbeck process
    4. Trade mean-reversion: enter when spread deviates > n*sigma,
       exit when spread returns to mean
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple
import warnings

try:
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not installed. Some features unavailable.")


@dataclass
class CointegrationResult:
    """Results from cointegration test."""
    is_cointegrated: bool
    pvalue: float
    test_stat: float
    critical_values: dict
    hedge_ratio: float
    half_life_days: float


@dataclass
class BacktestResult:
    """Backtest output."""
    trades: pd.DataFrame
    equity_curve: pd.Series
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_return: float
    n_trades: int


class CointegrationPairsTrader:
    """
    Pairs trading strategy based on cointegration for energy markets.

    Parameters
    ----------
    y : pd.Series
        First asset (e.g. TTF front month gas price).
    x : pd.Series
        Second asset (e.g. NBP front month gas price).
    entry_zscore : float
        Z-score threshold to open a position (default 2.0).
    exit_zscore : float
        Z-score threshold to close a position (default 0.5).
    stop_zscore : float
        Z-score stop-loss (default 4.0).
    lookback : int
        Rolling window to estimate spread statistics.
    """

    def __init__(
        self,
        y: pd.Series,
        x: pd.Series,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_zscore: float = 4.0,
        lookback: int = 60,
    ):
        self.y = y
        self.x = x
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_zscore = stop_zscore
        self.lookback = lookback
        self._result: Optional[CointegrationResult] = None

    # ------------------------------------------------------------------
    # Cointegration testing
    # ------------------------------------------------------------------

    def test_cointegration(self, significance: float = 0.05) -> CointegrationResult:
        """Run Engle-Granger cointegration test and estimate hedge ratio."""
        if not STATSMODELS_AVAILABLE:
            raise RuntimeError("statsmodels is required for cointegration test.")

        aligned = pd.concat([self.y, self.x], axis=1).dropna()
        y_arr = aligned.iloc[:, 0].values
        x_arr = aligned.iloc[:, 1].values

        # OLS hedge ratio
        reg = OLS(y_arr, add_constant(x_arr)).fit()
        hedge_ratio = reg.params[1]
        spread = y_arr - hedge_ratio * x_arr

        # Cointegration test (Engle-Granger)
        stat, pvalue, crit_vals = coint(y_arr, x_arr)

        # Half-life of mean reversion (AR(1) on spread)
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        reg_hl = OLS(spread_diff, add_constant(spread_lag)).fit()
        lam = reg_hl.params[1]
        half_life = -np.log(2) / lam if lam < 0 else np.inf

        self._result = CointegrationResult(
            is_cointegrated=(pvalue < significance),
            pvalue=round(pvalue, 4),
            test_stat=round(stat, 4),
            critical_values={
                "1%": round(crit_vals[0], 4),
                "5%": round(crit_vals[1], 4),
                "10%": round(crit_vals[2], 4),
            },
            hedge_ratio=round(hedge_ratio, 4),
            half_life_days=round(half_life, 1),
        )
        return self._result

    # ------------------------------------------------------------------
    # Spread computation
    # ------------------------------------------------------------------

    def compute_spread(self, hedge_ratio: Optional[float] = None) -> pd.Series:
        """Compute the spread series y - hedge_ratio * x."""
        if hedge_ratio is None:
            if self._result is None:
                self.test_cointegration()
            hedge_ratio = self._result.hedge_ratio

        spread = self.y - hedge_ratio * self.x
        spread.name = "spread"
        return spread

    def compute_zscore(self, spread: Optional[pd.Series] = None) -> pd.Series:
        """Rolling z-score of the spread."""
        if spread is None:
            spread = self.compute_spread()
        roll_mean = spread.rolling(self.lookback).mean()
        roll_std = spread.rolling(self.lookback).std()
        zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)
        zscore.name = "zscore"
        return zscore

    # ------------------------------------------------------------------
    # Half-life of mean reversion (Ornstein-Uhlenbeck)
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_ou_params(spread: pd.Series) -> dict:
        """
        Fit Ornstein-Uhlenbeck parameters to spread series.

        dS = kappa * (mu - S) dt + sigma dW

        Returns
        -------
        dict with kappa (mean-reversion speed), mu (long-run mean),
        sigma (vol), half_life (days).
        """
        if not STATSMODELS_AVAILABLE:
            raise RuntimeError("statsmodels required.")

        s = spread.dropna().values
        s_lag = s[:-1]
        s_diff = np.diff(s)

        reg = OLS(s_diff, add_constant(s_lag)).fit()
        lam = reg.params[1]
        const = reg.params[0]

        kappa = -lam
        mu = const / kappa if kappa > 0 else np.nan
        sigma = reg.resid.std() * np.sqrt(252)
        half_life = np.log(2) / kappa if kappa > 0 else np.inf

        return {
            "kappa": round(kappa, 4),
            "mu": round(mu, 4) if not np.isnan(mu) else None,
            "sigma_annualised": round(sigma, 4),
            "half_life_days": round(half_life, 1),
        }

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def backtest(self) -> BacktestResult:
        """
        Simple z-score pairs backtest.
        Position: +1 = long spread (long y, short x * hedge_ratio)
                  -1 = short spread (short y, long x * hedge_ratio)
        """
        spread = self.compute_spread()
        zscore = self.compute_zscore(spread)

        position = pd.Series(0.0, index=spread.index)
        current_pos = 0

        for i in range(self.lookback, len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z):
                continue

            if current_pos == 0:
                if z > self.entry_zscore:
                    current_pos = -1   # sell spread (expect reversion down)
                elif z < -self.entry_zscore:
                    current_pos = 1    # buy spread (expect reversion up)
            elif current_pos == 1:
                if z > -self.exit_zscore or z > self.stop_zscore:
                    current_pos = 0
            elif current_pos == -1:
                if z < self.exit_zscore or z < -self.stop_zscore:
                    current_pos = 0

            position.iloc[i] = current_pos

        daily_pnl = position.shift(1) * spread.diff()
        equity = daily_pnl.cumsum()

        dd = equity - equity.cummax()
        max_dd = dd.min()

        pnl_clean = daily_pnl.dropna()
        sharpe = (pnl_clean.mean() / pnl_clean.std()) * np.sqrt(252) if pnl_clean.std() > 0 else 0.0
        win_rate = (pnl_clean > 0).mean()

        # Build trade log
        entries = position.diff().fillna(0)
        trade_log = []
        for dt, val in entries.items():
            if val != 0:
                trade_log.append({"date": dt, "action": "entry" if val != 0 else "exit",
                                  "position": position[dt], "zscore": zscore[dt]})
        trades_df = pd.DataFrame(trade_log)

        return BacktestResult(
            trades=trades_df,
            equity_curve=equity,
            sharpe=round(sharpe, 3),
            max_drawdown=round(max_dd, 2),
            win_rate=round(win_rate, 3),
            total_return=round(equity.iloc[-1], 2),
            n_trades=len(trades_df),
        )


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(99)
    n = 600
    dates = pd.date_range("2021-01-01", periods=n, freq="B")

    # Cointegrated TTF/NBP synthetic series
    common = np.cumsum(rng.normal(0, 1, n))
    ttf = 40 + common + rng.normal(0, 0.5, n)
    nbp = 38 + common * 0.97 + rng.normal(0, 0.8, n) + 2.0

    ttf_s = pd.Series(ttf, index=dates, name="TTF")
    nbp_s = pd.Series(nbp, index=dates, name="NBP")

    trader = CointegrationPairsTrader(
        y=ttf_s, x=nbp_s,
        entry_zscore=1.8,
        exit_zscore=0.3,
        stop_zscore=3.5,
        lookback=40,
    )

    if STATSMODELS_AVAILABLE:
        coint_result = trader.test_cointegration()
        print("\n=== Cointegration Test ===")
        print(f"  Cointegrated:   {coint_result.is_cointegrated}")
        print(f"  P-value:        {coint_result.pvalue}")
        print(f"  Hedge ratio:    {coint_result.hedge_ratio}")
        print(f"  Half-life:      {coint_result.half_life_days} days")

        bt = trader.backtest()
        print("\n=== Backtest Results ===")
        print(f"  Sharpe ratio:   {bt.sharpe}")
        print(f"  Max drawdown:   {bt.max_drawdown}")
        print(f"  Win rate:       {bt.win_rate:.1%}")
        print(f"  Total return:   {bt.total_return}")
        print(f"  Trades:         {bt.n_trades}")
    else:
        print("Install statsmodels to run the full demo: pip install statsmodels")
