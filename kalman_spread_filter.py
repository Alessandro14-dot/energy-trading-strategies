"""
Kalman Filter — Dynamic Hedge Ratio for Energy Pairs
======================================================
Implements a Kalman filter to estimate a time-varying hedge ratio
between two cointegrated energy price series.

Motivation:
    In static cointegration (OLS), the hedge ratio β is fixed over the
    full sample. In energy markets, the relationship between two prices
    can shift over time due to:
        - Structural changes (new pipelines, capacity additions)
        - Regime changes (market coupling, regulatory shifts)
        - Seasonal dynamics (summer vs winter price relationships)
        - Fuel mix changes (renewable buildout affecting spark spread)

    The Kalman filter estimates β dynamically, adapting to structural
    changes while still exploiting mean reversion in the residual spread.

State-space model:
    Observation: y(t) = β(t) * x(t) + α(t) + ε(t)    ε ~ N(0, R)
    State:        β(t) = β(t-1) + δ(t)                 δ ~ N(0, Q)

    where:
        y(t)  = price of asset A (e.g. TTF)
        x(t)  = price of asset B (e.g. NCG)
        β(t)  = dynamic hedge ratio
        α(t)  = dynamic intercept (optional)
        R     = observation noise variance
        Q     = state (hedge ratio) evolution variance

    Kalman equations:
        Predict:  β_pred = β_t-1
                  P_pred = P_t-1 + Q

        Update:   y_hat  = α + β_pred * x(t)
                  e(t)   = y(t) - y_hat           (innovation / spread)
                  S(t)   = P_pred * x(t)^2 + R    (innovation variance)
                  K(t)   = P_pred * x(t) / S(t)   (Kalman gain)
                  β(t)   = β_pred + K(t) * e(t)   (updated state)
                  P(t)   = (1 - K(t)*x(t)) * P_pred

Trading logic:
    - The Kalman filter residual e(t) is the "spread" to trade
    - Unlike static spread, e(t) has constant variance → z-score stable
    - Enter when |e(t)| > entry_threshold * sqrt(S(t))
    - Exit when |e(t)| < exit_threshold * sqrt(S(t))

Advantages over static OLS:
    - Hedge ratio adapts to structural changes
    - Residual variance estimated in real-time
    - No look-ahead bias
    - Works better in non-stationary relationships

Applications in energy:
    - TTF vs NCG (dynamic pipeline relationship)
    - Power DE vs Power FR (changing interconnector flows)
    - Spark spread hedge ratio (efficiency curve)
    - Dark/spark spread relative value
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class KalmanConfig:
    """Configuration for the Kalman filter pairs strategy."""
    delta: float            = 1e-4      # state evolution variance (Q = delta * I)
    vt: float               = 1e-3      # observation noise variance (R)
    entry_multiplier: float = 1.5       # entry at ± entry_multiplier * sqrt(S)
    exit_multiplier: float  = 0.3       # exit  at ± exit_multiplier  * sqrt(S)
    stop_multiplier: float  = 4.0       # stop  at ± stop_multiplier  * sqrt(S)
    use_intercept: bool     = True      # estimate time-varying intercept α(t)
    min_periods: int        = 30        # minimum observations before trading


class KalmanPairsFilter:
    """
    Kalman filter estimator for dynamic hedge ratio.

    Estimates β(t) (and optionally α(t)) between two price series,
    returning the residual spread and its variance for trading signals.

    Parameters
    ----------
    y : pd.Series   Dependent variable (price A).
    x : pd.Series   Independent variable (price B).
    config : KalmanConfig
    """

    def __init__(
        self,
        y: pd.Series,
        x: pd.Series,
        config: Optional[KalmanConfig] = None,
    ):
        self.y   = y.rename("price_A")
        self.x   = x.rename("price_B")
        self.cfg = config or KalmanConfig()

        # State dimension: 1 (beta only) or 2 (alpha + beta)
        self.n_states = 2 if self.cfg.use_intercept else 1

    def run_filter(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the Kalman filter and return state estimates and innovations.

        Returns
        -------
        states : pd.DataFrame   Columns: beta, [alpha], P_beta, [P_alpha]
        innov  : pd.DataFrame   Columns: e (innovation), S (variance), zscore
        """
        n    = len(self.y)
        ns   = self.n_states
        cfg  = self.cfg

        # State vector θ = [β] or [α, β]
        theta = np.zeros(ns)
        P     = np.eye(ns) * 1.0      # initial state covariance
        Q     = np.eye(ns) * cfg.delta # process noise covariance
        R     = cfg.vt                 # observation noise variance

        betas   = np.full(n, np.nan)
        alphas  = np.full(n, np.nan) if ns == 2 else None
        e_arr   = np.full(n, np.nan)
        S_arr   = np.full(n, np.nan)
        z_arr   = np.full(n, np.nan)
        P_beta  = np.full(n, np.nan)

        y_arr = self.y.values
        x_arr = self.x.values

        for t in range(n):
            xt = x_arr[t]
            yt = y_arr[t]

            if np.isnan(xt) or np.isnan(yt):
                continue

            # Observation vector
            if ns == 2:
                F = np.array([1.0, xt])   # [1, x] → α + β*x
            else:
                F = np.array([xt])         # [x]  → β*x

            # Predict
            theta_pred = theta          # random walk: θ_t = θ_t-1
            P_pred     = P + Q

            # Innovation
            y_hat = F @ theta_pred
            e     = yt - y_hat
            S     = F @ P_pred @ F + R
            zscore = e / np.sqrt(max(S, 1e-12))

            # Update (Kalman gain)
            K     = P_pred @ F / S
            theta = theta_pred + K * e
            P     = (np.eye(ns) - np.outer(K, F)) @ P_pred

            # Store
            if ns == 2:
                alphas[t] = theta[0]
                betas[t]  = theta[1]
                P_beta[t] = P[1, 1]
            else:
                betas[t]  = theta[0]
                P_beta[t] = P[0, 0]

            e_arr[t] = e
            S_arr[t] = S
            z_arr[t] = zscore

        idx = self.y.index
        states_dict = {"beta": betas, "P_beta": P_beta}
        if alphas is not None:
            states_dict["alpha"] = alphas

        states = pd.DataFrame(states_dict, index=idx)
        innov  = pd.DataFrame({
            "innovation": e_arr,
            "S_variance": S_arr,
            "zscore":     z_arr,
        }, index=idx)

        return states, innov

    def compute_spread(self) -> pd.Series:
        """
        Compute the Kalman-filtered spread (innovation series).
        This is the residual after removing the dynamic hedge.
        """
        _, innov = self.run_filter()
        return innov["innovation"].rename("kalman_spread")


class KalmanPairsStrategy:
    """
    Pairs trading strategy using Kalman filter dynamic hedge ratio.

    Parameters
    ----------
    y      : pd.Series   Price of asset A.
    x      : pd.Series   Price of asset B.
    config : KalmanConfig
    """

    def __init__(
        self,
        y: pd.Series,
        x: pd.Series,
        config: Optional[KalmanConfig] = None,
    ):
        self.y      = y
        self.x      = x
        self.cfg    = config or KalmanConfig()
        self.filter = KalmanPairsFilter(y, x, config)
        self.results: Optional[pd.DataFrame] = None
        self._states: Optional[pd.DataFrame] = None
        self._innov:  Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Generate trading signals from Kalman filter residuals."""
        states, innov = self.filter.run_filter()
        self._states  = states
        self._innov   = innov

        zscore   = innov["zscore"]
        position = pd.Series(0.0, index=self.y.index)
        current  = 0

        for i in range(self.cfg.min_periods, len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z):
                continue
            if current == 0:
                if z < -self.cfg.entry_multiplier:
                    current =  1    # spread below → long A
                elif z > self.cfg.entry_multiplier:
                    current = -1    # spread above → short A
            elif current == 1:
                if z >= -self.cfg.exit_multiplier or z <= -self.cfg.stop_multiplier:
                    current = 0
            elif current == -1:
                if z <=  self.cfg.exit_multiplier or z >=  self.cfg.stop_multiplier:
                    current = 0
            position.iloc[i] = current

        spread    = innov["innovation"]
        daily_pnl = position.shift(1) * spread.diff()

        self.results = pd.DataFrame({
            "price_A":   self.y,
            "price_B":   self.x,
            "beta":      states["beta"],
            "alpha":     states.get("alpha", np.nan),
            "spread":    spread,
            "S_var":     innov["S_variance"],
            "zscore":    zscore,
            "position":  position,
            "daily_pnl": daily_pnl,
            "cum_pnl":   daily_pnl.cumsum(),
        })
        return self.results

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def hedge_ratio_drift(self) -> dict:
        """Statistics on how much the hedge ratio has drifted over time."""
        if self._states is None:
            self.run()
        beta = self._states["beta"].dropna()
        return {
            "beta_mean":   round(beta.mean(), 4),
            "beta_std":    round(beta.std(), 4),
            "beta_min":    round(beta.min(), 4),
            "beta_max":    round(beta.max(), 4),
            "beta_drift":  round(beta.iloc[-1] - beta.iloc[0], 4),
        }

    def summary(self) -> dict:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        pnl    = self.results["daily_pnl"].dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        dd     = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        return {
            **self.hedge_ratio_drift(),
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

    def plot(self, figsize=(14, 13)) -> plt.Figure:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        df  = self.results
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle("Kalman Filter — Dynamic Hedge Ratio Pairs Strategy",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["price_A"], label="Price A", color="#1565c0", lw=1.0)
        ax2 = ax.twinx()
        ax2.plot(df.index, df["price_B"], label="Price B", color="#c62828", lw=1.0, alpha=0.8)
        ax.set_ylabel("Price A"); ax2.set_ylabel("Price B")
        ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
        ax.set_title("Asset Prices", fontsize=10)

        ax = axes[1]
        ax.plot(df.index, df["beta"], color="#e65100", lw=1.0, label="Dynamic β(t)")
        if "alpha" in df.columns and not df["alpha"].isna().all():
            ax2b = ax.twinx()
            ax2b.plot(df.index, df["alpha"], color="#9c27b0", lw=0.8, ls="--", label="Dynamic α(t)")
            ax2b.set_ylabel("α"); ax2b.legend(loc="upper right", fontsize=8)
        ax.legend(loc="upper left", fontsize=8); ax.set_ylabel("β")
        ax.set_title("Kalman Filter: Dynamic Hedge Ratio β(t)", fontsize=10)

        ax = axes[2]
        ax.plot(df.index, df["zscore"], color="#555", lw=0.8)
        ax.axhline(self.cfg.entry_multiplier,  color="red",   lw=0.8, ls="--",
                   label=f"+{self.cfg.entry_multiplier}σ")
        ax.axhline(-self.cfg.entry_multiplier, color="green", lw=0.8, ls="--",
                   label=f"-{self.cfg.entry_multiplier}σ")
        ax.axhline(0, color="black", lw=0.3)
        longs  = df[df["position"] == 1]
        shorts = df[df["position"] == -1]
        ax.scatter(longs.index,  longs["zscore"],  s=14, color="green", zorder=5, label="Long A")
        ax.scatter(shorts.index, shorts["zscore"], s=14, color="red",   zorder=5, label="Short A")
        ax.legend(fontsize=8); ax.set_ylabel("Innovation Z-score")
        ax.set_title("Kalman Spread Z-score & Signals", fontsize=10)

        ax = axes[3]
        ax.fill_between(df.index, df["cum_pnl"], 0,
                        where=df["cum_pnl"] >= 0, color="green", alpha=0.4)
        ax.fill_between(df.index, df["cum_pnl"], 0,
                        where=df["cum_pnl"] < 0,  color="red",   alpha=0.4)
        ax.plot(df.index, df["cum_pnl"], color="black", lw=0.8)
        ax.set_ylabel("Cumul. P&L"); ax.set_title("Cumulative P&L", fontsize=10)
        ax.set_xlabel("Date")

        plt.tight_layout()
        return fig


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng   = np.random.default_rng(99)
    n     = 800
    dates = pd.date_range("2021-01-01", periods=n, freq="B")

    # Slowly drifting hedge ratio
    beta_true = 0.95 + np.cumsum(rng.normal(0, 0.001, n))
    common    = np.cumsum(rng.normal(0, 1.5, n))
    x_prices  = 50 + common + rng.normal(0, 0.5, n)
    y_prices  = 2.0 + beta_true * x_prices + rng.normal(0, 0.8, n)

    y = pd.Series(y_prices, index=dates, name="TTF")
    x = pd.Series(x_prices, index=dates, name="NCG")

    cfg   = KalmanConfig(delta=1e-4, vt=1e-3,
                         entry_multiplier=1.5, exit_multiplier=0.3,
                         use_intercept=True, min_periods=30)
    strat = KalmanPairsStrategy(y=y, x=x, config=cfg)

    results = strat.run()
    stats   = strat.summary()

    print("\n=== Kalman Filter Pairs — Summary ===")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    fig = strat.plot()
    fig.savefig("kalman_spread_filter.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → kalman_spread_filter.png")
