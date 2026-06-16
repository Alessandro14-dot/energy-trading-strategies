"""
Ornstein-Uhlenbeck Mean Reversion Model
=========================================
Fits the Ornstein-Uhlenbeck (OU) process to energy spread series
and generates optimal entry/exit signals based on OU parameters.

Theory:
    The OU process models a mean-reverting stochastic process:

        dS(t) = κ(μ - S(t))dt + σ dW(t)

    where:
        κ (kappa)  = mean-reversion speed  [1/time]
        μ (mu)     = long-run mean
        σ (sigma)  = instantaneous volatility
        W(t)       = Wiener process (Brownian motion)

    Key derived quantities:
        Half-life  = ln(2) / κ          [days to revert halfway]
        Std dev of stationary dist = σ / sqrt(2κ)
        Optimal entry band = ± c * σ_eq  where c is chosen to maximise Sharpe

    Why it matters for energy trading:
        - Spark spreads, hub spreads, calendar spreads all exhibit mean reversion
        - OU parameters tell you HOW FAST and HOW FAR to let a spread run
        - Short half-life → trade intraday or short-term
        - Long half-life → position can take weeks to monetise
        - σ_eq → natural risk unit for position sizing

    Discrete-time estimation (AR(1)):
        S(t) = a + b*S(t-1) + ε(t)
        κ = -ln(b) / dt
        μ = a / (1 - b)
        σ = std(ε) / sqrt(dt)

    Optimal threshold (Bertram 2010):
        Maximise expected return per unit time:
        Entry at ±m, exit at 0
        Optimal m ≈ σ_eq * c  (c ≈ 1.0 to 1.5 for typical Sharpe)

Applications in energy:
    - Spark spread (CCGT margin)
    - Hub spread (TTF-NCG, TTF-NBP)
    - Dark spread (coal margin)
    - Cross-border power spreads
    - Any stationary spread after cointegration test
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from scipy.optimize import minimize_scalar

try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False


# ─── OU parameter estimation ─────────────────────────────────────────────────

@dataclass
class OUParams:
    """Estimated parameters of the Ornstein-Uhlenbeck process."""
    kappa: float            # mean-reversion speed [1/day]
    mu: float               # long-run mean
    sigma: float            # instantaneous volatility [per sqrt(day)]
    sigma_eq: float         # equilibrium (stationary) std dev
    half_life_days: float   # days to revert halfway to mean
    r_squared: float        # AR(1) fit quality


def estimate_ou_params(
    series: pd.Series,
    dt: float = 1.0,
) -> OUParams:
    """
    Estimate OU parameters from a time series via AR(1) OLS regression.

    Parameters
    ----------
    series : pd.Series   Spread or price series.
    dt     : float       Time step in days (default 1 = daily data).

    Returns
    -------
    OUParams dataclass with all OU parameters.
    """
    s       = series.dropna().values
    s_lag   = s[:-1]
    s_curr  = s[1:]

    if not STATSMODELS_OK:
        # Fallback: numpy least squares
        X = np.column_stack([np.ones_like(s_lag), s_lag])
        coeffs, _, _, _ = np.linalg.lstsq(X, s_curr, rcond=None)
        a, b = coeffs
        resid = s_curr - (a + b * s_lag)
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((s_curr - s_curr.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        reg = OLS(s_curr, add_constant(s_lag)).fit()
        a, b = reg.params
        resid = reg.resid
        r2    = reg.rsquared

    # OU parameter extraction
    if b >= 1.0 or b <= 0.0:
        # Non-stationary → set kappa to near-zero
        kappa    = 1e-6
        mu       = float(np.mean(s))
        sigma    = float(np.std(resid)) / np.sqrt(dt)
        sigma_eq = sigma / np.sqrt(2 * kappa + 1e-9)
        hl       = np.inf
    else:
        kappa    = -np.log(b) / dt
        mu       = a / (1 - b)
        sigma    = float(np.std(resid)) / np.sqrt(dt)
        sigma_eq = sigma / np.sqrt(2 * kappa)
        hl       = np.log(2) / kappa

    return OUParams(
        kappa          = round(kappa, 6),
        mu             = round(mu, 4),
        sigma          = round(sigma, 4),
        sigma_eq       = round(sigma_eq, 4),
        half_life_days = round(hl, 2) if hl != np.inf else np.inf,
        r_squared      = round(r2, 4),
    )


# ─── Optimal threshold ───────────────────────────────────────────────────────

def optimal_entry_threshold(
    ou_params: OUParams,
    transaction_cost: float = 0.0,
    n_grid: int = 200,
) -> Dict[str, float]:
    """
    Find the optimal entry threshold m that maximises expected Sharpe ratio.

    Uses Bertram (2010) framework:
        Expected return per trade ∝ m (wider entry = more profit per trade)
        Expected time per trade ∝ 1/κ * f(m/σ_eq) (wider = longer to trigger)
        Optimal m balances these two effects.

    For practical use: optimal m ≈ 0.75 * σ_eq to 1.5 * σ_eq.
    With transaction costs, optimal m increases (need bigger edge to cover TC).
    """
    se = ou_params.sigma_eq
    k  = ou_params.kappa

    if se <= 0 or k <= 0:
        return {"optimal_m": se, "optimal_m_sigma_units": 1.0, "expected_trades_per_year": 0}

    # Simplified expected Sharpe approximation on grid
    ms = np.linspace(0.1 * se, 3.0 * se, n_grid)

    def expected_sharpe(m):
        # Expected profit per trade (net of TC)
        profit = 2 * (m - transaction_cost)
        if profit <= 0:
            return -np.inf
        # Expected time to complete trade (approximate OU first passage time)
        # E[T] ≈ (1/κ) * (m/σ_eq)^2  — simplified
        expected_time = (1 / k) * (m / se) ** 2
        return profit / (se * np.sqrt(expected_time + 1e-9))

    sharpes = np.array([expected_sharpe(m) for m in ms])
    best_idx = np.argmax(sharpes)

    return {
        "optimal_m":                round(ms[best_idx], 4),
        "optimal_m_sigma_units":    round(ms[best_idx] / se, 3),
        "max_expected_sharpe":      round(sharpes[best_idx], 3),
        "expected_trades_per_year": round(252 * k / (ms[best_idx] / se) ** 2, 1),
    }


# ─── OU Trading Strategy ─────────────────────────────────────────────────────

@dataclass
class OUStrategyConfig:
    """Configuration for OU mean-reversion strategy."""
    lookback: int           = 60        # rolling window for parameter estimation
    entry_band: float       = 1.5       # entry at ±entry_band * sigma_eq
    exit_band: float        = 0.2       # exit at ±exit_band * sigma_eq
    stop_band: float        = 3.5       # stop at ±stop_band * sigma_eq
    use_optimal_band: bool  = False     # override entry_band with optimal threshold
    transaction_cost: float = 0.0      # round-trip cost [same units as spread]
    reestimate_freq: int    = 20        # re-estimate OU params every N days


class OUMeanReversionStrategy:
    """
    Mean-reversion strategy using Ornstein-Uhlenbeck process.

    Parameters
    ----------
    spread : pd.Series       Spread to trade (must be stationary / cointegrated).
    config : OUStrategyConfig
    """

    def __init__(
        self,
        spread: pd.Series,
        config: Optional[OUStrategyConfig] = None,
    ):
        self.spread  = spread.rename("spread")
        self.cfg     = config or OUStrategyConfig()
        self.results: Optional[pd.DataFrame] = None
        self._ou_params_history: Dict[pd.Timestamp, OUParams] = {}

    # ------------------------------------------------------------------
    # Rolling OU estimation
    # ------------------------------------------------------------------

    def rolling_ou_params(self) -> pd.DataFrame:
        """
        Estimate OU parameters on a rolling basis.
        Re-estimates every `reestimate_freq` days.
        """
        records = []
        idx     = self.spread.index
        lb      = self.cfg.lookback

        for i in range(lb, len(self.spread), self.cfg.reestimate_freq):
            window = self.spread.iloc[i - lb: i]
            try:
                p = estimate_ou_params(window)
            except Exception:
                continue
            self._ou_params_history[idx[i]] = p
            records.append({
                "date":       idx[i],
                "kappa":      p.kappa,
                "mu":         p.mu,
                "sigma":      p.sigma,
                "sigma_eq":   p.sigma_eq,
                "half_life":  p.half_life_days,
                "r_squared":  p.r_squared,
            })

        return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()

    def _interpolate_params(self, ou_df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate OU params to daily frequency."""
        if ou_df.empty:
            return pd.DataFrame()
        return ou_df.reindex(self.spread.index).interpolate("time").ffill().bfill()

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Generate trading signals using rolling OU parameters.
        Entry at ±entry_band * sigma_eq, exit at ±exit_band * sigma_eq.
        """
        ou_df   = self.rolling_ou_params()
        ou_full = self._interpolate_params(ou_df)

        spread  = self.spread
        position = pd.Series(0.0, index=spread.index)
        current  = 0

        # Rolling mean as mu proxy if OU estimation not available
        roll_mu  = spread.rolling(self.cfg.lookback).mean()
        roll_seq = spread.rolling(self.cfg.lookback).std()   # fallback sigma_eq

        for i in range(self.cfg.lookback, len(spread)):
            dt   = spread.index[i]
            s    = spread.iloc[i]

            # Get sigma_eq and mu
            if not ou_full.empty and dt in ou_full.index:
                mu_i  = ou_full.loc[dt, "mu"]
                seq_i = ou_full.loc[dt, "sigma_eq"]
            else:
                mu_i  = roll_mu.iloc[i]
                seq_i = roll_seq.iloc[i]

            if np.isnan(mu_i) or np.isnan(seq_i) or seq_i <= 0:
                continue

            deviation = s - mu_i
            entry_th  = self.cfg.entry_band * seq_i
            exit_th   = self.cfg.exit_band  * seq_i
            stop_th   = self.cfg.stop_band  * seq_i

            if current == 0:
                if deviation < -entry_th:
                    current =  1    # spread below mean → long
                elif deviation > entry_th:
                    current = -1    # spread above mean → short
            elif current == 1:
                if deviation >= -exit_th or deviation <= -stop_th:
                    current = 0
            elif current == -1:
                if deviation <= exit_th or deviation >= stop_th:
                    current = 0

            position.iloc[i] = current

        daily_pnl = position.shift(1) * spread.diff()

        # Build results with OU stats
        mu_series  = roll_mu.copy()
        seq_series = roll_seq.copy()
        if not ou_full.empty:
            mu_series  = ou_full.get("mu",       mu_series)
            seq_series = ou_full.get("sigma_eq", seq_series)

        self.results = pd.DataFrame({
            "spread":       spread,
            "ou_mu":        mu_series,
            "ou_sigma_eq":  seq_series,
            "upper_band":   mu_series + self.cfg.entry_band * seq_series,
            "lower_band":   mu_series - self.cfg.entry_band * seq_series,
            "position":     position,
            "daily_pnl":    daily_pnl,
            "cum_pnl":      daily_pnl.cumsum(),
        })
        return self.results

    # ------------------------------------------------------------------
    # Full OU diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict:
        """Full OU diagnostics on the entire spread series."""
        p    = estimate_ou_params(self.spread)
        opt  = optimal_entry_threshold(p, self.cfg.transaction_cost)
        return {
            "kappa":                 p.kappa,
            "mu":                    p.mu,
            "sigma":                 p.sigma,
            "sigma_eq":              p.sigma_eq,
            "half_life_days":        p.half_life_days,
            "ar1_r_squared":         p.r_squared,
            **opt,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        pnl    = self.results["daily_pnl"].dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        dd     = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        diag   = self.diagnostics()
        return {
            **diag,
            "entry_band_used":   self.cfg.entry_band,
            "total_pnl":         round(pnl.sum(), 2),
            "sharpe_ratio":      round(sharpe, 3),
            "max_drawdown":      round(dd, 2),
            "win_rate":          round((pnl > 0).mean(), 3),
            "n_long":            int((self.results["position"] == 1).sum()),
            "n_short":           int((self.results["position"] == -1).sum()),
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, figsize=(14, 11)) -> plt.Figure:
        if self.results is None:
            raise RuntimeError("Call run() first.")
        df  = self.results
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        fig.suptitle("Ornstein-Uhlenbeck Mean Reversion Strategy",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["spread"],      color="#333",   lw=0.9, label="Spread")
        ax.plot(df.index, df["ou_mu"],       color="#9c27b0", lw=0.9, ls="--", label="OU Mean (μ)")
        ax.plot(df.index, df["upper_band"],  color="red",    lw=0.8, ls=":",   label=f"+{self.cfg.entry_band}σ_eq")
        ax.plot(df.index, df["lower_band"],  color="green",  lw=0.8, ls=":",   label=f"-{self.cfg.entry_band}σ_eq")
        ax.fill_between(df.index, df["upper_band"], df["lower_band"],
                        alpha=0.06, color="#9c27b0")
        longs  = df[df["position"] == 1]
        shorts = df[df["position"] == -1]
        ax.scatter(longs.index,  longs["spread"],  s=18, color="green", zorder=5, label="Long")
        ax.scatter(shorts.index, shorts["spread"], s=18, color="red",   zorder=5, label="Short")
        ax.legend(fontsize=8); ax.set_ylabel("Spread"); ax.set_title("Spread & OU Bands", fontsize=10)

        ax = axes[1]
        ax.fill_between(df.index, df["position"], 0,
                        where=df["position"] == 1,  color="green", alpha=0.5, label="Long")
        ax.fill_between(df.index, df["position"], 0,
                        where=df["position"] == -1, color="red",   alpha=0.5, label="Short")
        ax.axhline(0, color="black", lw=0.4)
        ax.legend(fontsize=8); ax.set_ylabel("Position")
        ax.set_title("Position (Long / Short)", fontsize=10)

        ax = axes[2]
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
    rng   = np.random.default_rng(55)
    n     = 800
    dates = pd.date_range("2021-01-01", periods=n, freq="B")

    # Simulate OU process: TTF-NCG spread
    kappa_true, mu_true, sigma_true = 0.15, 0.5, 0.3
    s = [mu_true]
    for _ in range(n - 1):
        ds = kappa_true * (mu_true - s[-1]) + sigma_true * rng.normal()
        s.append(s[-1] + ds)
    spread = pd.Series(s, index=dates, name="TTF_NCG_spread")

    # Full diagnostics
    params = estimate_ou_params(spread)
    print("\n=== OU Parameter Estimation ===")
    print(f"  True  κ = {kappa_true:.4f}  μ = {mu_true:.4f}  σ = {sigma_true:.4f}")
    print(f"  Est.  κ = {params.kappa:.4f}  μ = {params.mu:.4f}  σ = {params.sigma:.4f}")
    print(f"  Half-life: {params.half_life_days:.1f} days  |  σ_eq: {params.sigma_eq:.4f}")

    opt = optimal_entry_threshold(params)
    print(f"\n=== Optimal Entry Threshold ===")
    for k, v in opt.items():
        print(f"  {k:35s}: {v}")

    cfg   = OUStrategyConfig(lookback=60, entry_band=1.5, exit_band=0.2)
    strat = OUMeanReversionStrategy(spread=spread, config=cfg)
    strat.run()
    stats = strat.summary()

    print("\n=== OU Strategy — Summary ===")
    for k, v in stats.items():
        print(f"  {k:35s}: {v}")

    fig = strat.plot()
    fig.savefig("ou_mean_reversion.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → ou_mean_reversion.png")
