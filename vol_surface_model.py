"""
Implied Volatility Surface Model — Energy Options
===================================================
Fits, analyses, and trades the implied volatility surface
for power and gas options markets.

Market context:
    European energy options are traded on:
        - EEX (European Energy Exchange): Power DE, FR options
        - ICE: TTF gas options, NBP options, Brent (cross-asset)
        - CME: Henry Hub gas options
        - OTC: bespoke structures via broker markets

    The vol surface has two dimensions:
        1. TENOR (expiry): from 1 month to 3+ years
        2. MONEYNESS (strike): expressed as delta or % OTM

    Key energy vol surface features:
        - SKEW: energy often has positive skew (upside vol > downside)
          because price spikes are more common than crashes
        - TERM STRUCTURE: short-dated vol > long-dated vol (mean-reverting)
        - SEASONAL EFFECTS: winter options more expensive than summer
        - SPIKE RISK PREMIUM: extreme vol events priced via heavy tails

    Vol surface models:
        - SABR: industry standard for energy options
        - SVI (Stochastic Volatility Inspired): arbitrage-free parametric
        - Heston: stochastic vol with mean reversion
        - Local vol: Dupire equation from market prices

    Key vol metrics:
        - ATM vol:       at-the-money implied vol
        - Risk reversal: 25-delta call vol - 25-delta put vol (skew proxy)
        - Butterfly:     (25d call + 25d put)/2 - ATM vol (convexity)
        - Vol of vol:    vol surface stability measure

Trading strategies:
    1. Vol surface fitting and arbitrage detection
    2. ATM vol mean reversion
    3. Skew trading (risk reversals)
    4. Term structure trading (calendar vol spreads)
    5. Realised vs implied vol (variance swaps proxy)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from scipy.stats import norm
from scipy.optimize import brentq, minimize


# ─── Black-76 option pricing ─────────────────────────────────────────────────

def black76_price(
    F: float,       # forward price
    K: float,       # strike
    T: float,       # time to expiry [years]
    r: float,       # risk-free rate
    sigma: float,   # implied vol
    option_type: str = "call",
) -> float:
    """Black-76 model for options on futures (standard for energy)."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (F - K) if option_type == "call" else (K - F))
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df = np.exp(-r * T)
    if option_type == "call":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_vega(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega of a Black-76 option."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return F * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)


def implied_vol_black76(
    F: float, K: float, T: float, r: float,
    market_price: float, option_type: str = "call",
    tol: float = 1e-6, max_iter: int = 100,
) -> Optional[float]:
    """
    Newton-Raphson implied vol solver for Black-76.
    Returns None if solver fails to converge.
    """
    intrinsic = max(0.0, (F - K) if option_type == "call" else (K - F))
    if market_price <= intrinsic + 1e-8:
        return None

    sigma = 0.30  # initial guess
    for _ in range(max_iter):
        price = black76_price(F, K, T, r, sigma, option_type)
        vega  = black76_vega(F, K, T, r, sigma)
        if abs(vega) < 1e-12:
            break
        sigma_new = sigma - (price - market_price) / vega
        sigma_new = max(1e-4, min(sigma_new, 5.0))
        if abs(sigma_new - sigma) < tol:
            return sigma_new
        sigma = sigma_new
    try:
        result = brentq(
            lambda s: black76_price(F, K, T, r, s, option_type) - market_price,
            1e-4, 5.0, xtol=tol,
        )
        return result
    except Exception:
        return None


# ─── SVI parametric vol surface ───────────────────────────────────────────────

def svi_vol(k: float, a: float, b: float, rho: float, m: float, sigma: float) -> float:
    """
    SVI (Stochastic Volatility Inspired) parametric total variance.
    w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    Implied vol = sqrt(w(k) / T)

    Parameters:
        k     = log-moneyness = log(K/F)
        a     = vertical translation (base variance)
        b     = slope/width parameter
        rho   = correlation/skew parameter [-1, 1]
        m     = horizontal translation (ATM location)
        sigma = curvature parameter > 0
    """
    total_var = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    return max(total_var, 1e-8)


def fit_svi(
    log_moneyness: np.ndarray,
    market_vols: np.ndarray,
    T: float,
) -> Dict[str, float]:
    """
    Fit SVI parameters to a slice of market implied vols.
    Minimises sum of squared errors between model and market total variance.
    """
    market_w = (market_vols ** 2) * T   # total variance

    def objective(params):
        a, b, rho, m, sigma = params
        if b < 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e10
        model_w = np.array([svi_vol(k, a, b, rho, m, sigma) for k in log_moneyness])
        return np.sum((model_w - market_w)**2)

    # Initial guess
    atm_var = float(np.mean(market_w))
    x0      = [atm_var * 0.8, 0.1, -0.3, 0.0, 0.1]
    bounds  = [(1e-6, 2.0), (1e-6, 1.0), (-0.999, 0.999), (-1.0, 1.0), (1e-4, 1.0)]

    try:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 1000})
        a, b, rho, m, sigma = res.x
        return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
                "fit_error": res.fun}
    except Exception:
        return {"a": atm_var, "b": 0.1, "rho": -0.3, "m": 0.0,
                "sigma": 0.1, "fit_error": np.nan}


# ─── Vol surface dataclass ────────────────────────────────────────────────────

@dataclass
class VolSurface:
    """
    Represents an implied volatility surface at a point in time.

    Attributes
    ----------
    tenors  : list of float   Option expiries in years (e.g. [0.083, 0.25, 0.5, 1.0])
    strikes : list of float   Strike prices [same units as forward]
    forward : float           Forward price of the underlying
    vols    : np.ndarray      Shape (n_tenors, n_strikes) of implied vols
    rate    : float           Risk-free rate
    """
    tenors:  List[float]
    strikes: List[float]
    forward: float
    vols:    np.ndarray       # shape: (n_tenors, n_strikes)
    rate:    float = 0.03
    date:    Optional[pd.Timestamp] = None

    def atm_vol(self, tenor_idx: int = 0) -> float:
        """ATM implied vol for a given tenor index."""
        F = self.forward
        K_idx = np.argmin(np.abs(np.array(self.strikes) - F))
        return self.vols[tenor_idx, K_idx]

    def skew(self, tenor_idx: int = 0) -> float:
        """
        Simple skew measure: vol(110% F) - vol(90% F).
        Positive skew = upside more expensive (energy typical).
        """
        strikes = np.array(self.strikes)
        F = self.forward
        idx_otm = np.argmin(np.abs(strikes - 1.10 * F))
        idx_otm_put = np.argmin(np.abs(strikes - 0.90 * F))
        return self.vols[tenor_idx, idx_otm] - self.vols[tenor_idx, idx_otm_put]

    def term_structure(self) -> pd.Series:
        """ATM vol by tenor."""
        return pd.Series(
            [self.atm_vol(i) for i in range(len(self.tenors))],
            index=[f"{T:.3f}Y" for T in self.tenors],
            name="atm_vol",
        )

    def log_moneyness_slice(self, tenor_idx: int) -> np.ndarray:
        """Log-moneyness array for a tenor: log(K/F)."""
        return np.log(np.array(self.strikes) / self.forward)

    def fit_svi_slice(self, tenor_idx: int) -> Dict[str, float]:
        """Fit SVI to a single tenor slice."""
        k = self.log_moneyness_slice(tenor_idx)
        v = self.vols[tenor_idx]
        T = self.tenors[tenor_idx]
        return fit_svi(k, v, T)


# ─── Vol surface builder ──────────────────────────────────────────────────────

def build_synthetic_surface(
    forward: float       = 80.0,
    tenors: List[float]  = None,
    strikes: List[float] = None,
    atm_vol: float       = 0.45,
    skew: float          = 0.08,
    term_slope: float    = -0.05,
    convexity: float     = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> VolSurface:
    """
    Build a realistic synthetic energy vol surface.

    Surface shape:
        - ATM vol decreasing with tenor (mean-reverting underlying)
        - Positive skew (upside > downside, typical energy)
        - Smile convexity
        - Small random noise
    """
    if tenors  is None: tenors  = [1/12, 3/12, 6/12, 1.0, 2.0]
    if strikes is None:
        strikes = [forward * k for k in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]]
    if rng is None: rng = np.random.default_rng(42)

    n_T = len(tenors)
    n_K = len(strikes)
    vols = np.zeros((n_T, n_K))

    for i, T in enumerate(tenors):
        # ATM vol: flat or slightly decreasing term structure
        atm_T = max(0.05, atm_vol + term_slope * T)
        for j, K in enumerate(strikes):
            lm = np.log(K / forward)
            # Skew component: positive lm (OTM call) → slightly higher
            skew_comp = skew * lm
            # Smile (convexity): vol smile
            smile_comp = convexity * lm**2
            vol = atm_T + skew_comp + smile_comp
            vol += rng.normal(0, 0.005)   # market noise
            vols[i, j] = max(0.05, vol)

    return VolSurface(
        tenors=tenors, strikes=strikes,
        forward=forward, vols=vols,
    )


# ─── Vol trading strategies ───────────────────────────────────────────────────

@dataclass
class VolStrategyConfig:
    """Configuration for vol surface trading strategies."""
    lookback: int           = 30
    entry_zscore: float     = 1.6
    exit_zscore: float      = 0.4
    stop_zscore: float      = 3.5
    risk_free_rate: float   = 0.03
    tenor_focus: float      = 0.25      # focus tenor for ATM vol signal [years]


class ATMVolStrategy:
    """
    Trades mean reversion of ATM implied volatility.

    Logic:
        When implied vol is unusually HIGH vs historical → sell vol (short straddle)
        When implied vol is unusually LOW  vs historical → buy vol  (long straddle)

    Instrument:
        ATM straddle = ATM call + ATM put
        Delta-neutral, pure vol play
        P&L ≈ (realised_vol² - implied_vol²) * vega

    Parameters
    ----------
    atm_vol_ts : pd.Series   Time series of ATM implied vol for target tenor.
    forward_ts : pd.Series   Forward price time series.
    realised_vol_ts : pd.Series  Realised vol time series (for comparison).
    config : VolStrategyConfig
    """

    def __init__(
        self,
        atm_vol_ts: pd.Series,
        forward_ts: pd.Series,
        realised_vol_ts: Optional[pd.Series] = None,
        config: Optional[VolStrategyConfig] = None,
    ):
        self.atm_vol  = atm_vol_ts.rename("atm_vol")
        self.forward  = forward_ts.rename("forward")
        self.realised = realised_vol_ts.rename("realised_vol") if realised_vol_ts is not None else None
        self.cfg      = config or VolStrategyConfig()
        self.results: Optional[pd.DataFrame] = None

    def vol_risk_premium(self) -> Optional[pd.Series]:
        """
        Vol risk premium: implied vol - realised vol.
        Positive = market paying premium for protection (sell vol opportunity).
        """
        if self.realised is None:
            return None
        vrp = self.atm_vol - self.realised
        vrp.name = "vol_risk_premium"
        return vrp

    def straddle_pnl_approx(
        self, position: pd.Series, T: float = 0.25
    ) -> pd.Series:
        """
        Approximate daily straddle P&L.
        Long straddle: gains when realised vol > implied vol.
        Short straddle: gains when realised vol < implied vol.

        Simplified: P&L ≈ position * (realised_var - implied_var) * F² * T
        """
        if self.realised is None:
            return position.shift(1) * self.atm_vol.diff() * (-1)
        daily_rv  = self.realised ** 2 / 252
        daily_iv  = self.atm_vol  ** 2 / 252
        var_diff  = daily_rv - daily_iv
        pnl       = position.shift(1) * var_diff * self.forward**2 * 0.001
        pnl.name  = "straddle_pnl"
        return pnl

    def run(self) -> pd.DataFrame:
        """Generate vol trading signals based on ATM vol z-score."""
        vol      = self.atm_vol
        roll_m   = vol.rolling(self.cfg.lookback).mean()
        roll_s   = vol.rolling(self.cfg.lookback).std()
        zscore   = (vol - roll_m) / roll_s.replace(0, np.nan)
        vrp      = self.vol_risk_premium()

        position = pd.Series(0.0, index=vol.index)
        current  = 0

        for i in range(self.cfg.lookback, len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z): continue
            if current == 0:
                if z > self.cfg.entry_zscore:
                    current = -1    # vol high → sell vol (short straddle)
                elif z < -self.cfg.entry_zscore:
                    current =  1    # vol low  → buy vol  (long straddle)
            elif current == -1:
                if z <= self.cfg.exit_zscore or z <= -self.cfg.stop_zscore:
                    current = 0
            elif current == 1:
                if z >= -self.cfg.exit_zscore or z >= self.cfg.stop_zscore:
                    current = 0
            position.iloc[i] = current

        daily_pnl = self.straddle_pnl_approx(position, T=self.cfg.tenor_focus)

        self.results = pd.DataFrame({
            "atm_vol":    vol,
            "forward":    self.forward,
            "realised":   self.realised if self.realised is not None else np.nan,
            "vrp":        vrp if vrp is not None else np.nan,
            "roll_mean":  roll_m,
            "zscore":     zscore,
            "position":   position,
            "daily_pnl":  daily_pnl,
            "cum_pnl":    daily_pnl.cumsum(),
        })
        return self.results

    def summary(self) -> dict:
        if self.results is None: raise RuntimeError("Call run() first.")
        pnl    = self.results["daily_pnl"].dropna()
        vol    = self.atm_vol.dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        dd     = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        return {
            "avg_atm_vol":      round(vol.mean(), 4),
            "std_atm_vol":      round(vol.std(), 4),
            "min_atm_vol":      round(vol.min(), 4),
            "max_atm_vol":      round(vol.max(), 4),
            "avg_vrp":          round(self.results["vrp"].mean(), 4) if not self.results["vrp"].isna().all() else "N/A",
            "total_pnl":        round(pnl.sum(), 4),
            "sharpe_ratio":     round(sharpe, 3),
            "max_drawdown":     round(dd, 4),
            "win_rate":         round((pnl > 0).mean(), 3),
            "n_long_vol":       int((self.results["position"] == 1).sum()),
            "n_short_vol":      int((self.results["position"] == -1).sum()),
        }

    def plot(self, figsize=(14, 12)) -> plt.Figure:
        if self.results is None: raise RuntimeError("Call run() first.")
        df  = self.results
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle("ATM Implied Vol Mean Reversion Strategy", fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["atm_vol"] * 100,  label="Implied Vol (%)", color="#c62828", lw=1.0)
        if not df["realised"].isna().all():
            ax.plot(df.index, df["realised"] * 100, label="Realised Vol (%)", color="#1565c0",
                    lw=0.9, ls="--", alpha=0.8)
        ax.plot(df.index, df["roll_mean"] * 100, label="Rolling Mean", color="#888", lw=0.8, ls=":")
        ax.legend(fontsize=8); ax.set_ylabel("Vol (%)"); ax.set_title("ATM Implied vs Realised Vol", fontsize=10)

        ax = axes[1]
        if not df["vrp"].isna().all():
            ax.fill_between(df.index, df["vrp"] * 100, 0,
                            where=df["vrp"] > 0, color="red",   alpha=0.4, label="Vol Premium (sell)")
            ax.fill_between(df.index, df["vrp"] * 100, 0,
                            where=df["vrp"] < 0, color="green", alpha=0.4, label="Vol Discount (buy)")
            ax.axhline(0, color="black", lw=0.5)
            ax.legend(fontsize=8); ax.set_ylabel("VRP (pp)")
        ax.set_title("Vol Risk Premium (Implied - Realised)", fontsize=10)

        ax = axes[2]
        ax.plot(df.index, df["zscore"], color="#555", lw=0.8)
        ax.axhline(self.cfg.entry_zscore,  color="red",   lw=0.8, ls="--", label="Sell vol")
        ax.axhline(-self.cfg.entry_zscore, color="green", lw=0.8, ls="--", label="Buy vol")
        ax.axhline(0, color="black", lw=0.3)
        sells = df[df["position"] == -1]
        buys  = df[df["position"] ==  1]
        ax.scatter(sells.index, sells["zscore"], s=14, color="red",   zorder=5)
        ax.scatter(buys.index,  buys["zscore"],  s=14, color="green", zorder=5)
        ax.legend(fontsize=8); ax.set_ylabel("Z-score"); ax.set_title("Signal", fontsize=10)

        ax = axes[3]
        ax.fill_between(df.index, df["cum_pnl"], 0,
                        where=df["cum_pnl"] >= 0, color="green", alpha=0.4)
        ax.fill_between(df.index, df["cum_pnl"], 0,
                        where=df["cum_pnl"] < 0,  color="red",   alpha=0.4)
        ax.plot(df.index, df["cum_pnl"], color="black", lw=0.8)
        ax.set_ylabel("P&L (approx)"); ax.set_title("Cumulative P&L", fontsize=10)
        ax.set_xlabel("Date")

        plt.tight_layout()
        return fig


# ─── Vol surface plotting utility ────────────────────────────────────────────

def plot_vol_surface(surface: VolSurface, figsize=(14, 6)) -> plt.Figure:
    """3D plot of the vol surface."""
    fig = plt.figure(figsize=figsize)

    # 2D slice plot (term structure + smile)
    ax1 = fig.add_subplot(121)
    for i, T in enumerate(surface.tenors):
        lm = surface.log_moneyness_slice(i) * 100
        ax1.plot(lm, surface.vols[i] * 100, label=f"T={T:.2f}Y", lw=1.2)
    ax1.axvline(0, color="black", lw=0.5, ls=":")
    ax1.set_xlabel("Log-moneyness (%)"); ax1.set_ylabel("Implied Vol (%)")
    ax1.set_title("Vol Smile by Tenor"); ax1.legend(fontsize=8)

    # ATM term structure
    ax2 = fig.add_subplot(122)
    ts = surface.term_structure()
    ax2.plot(surface.tenors, ts.values * 100, "o-", color="#c62828", lw=1.5, ms=7)
    ax2.set_xlabel("Tenor (years)"); ax2.set_ylabel("ATM Implied Vol (%)")
    ax2.set_title("ATM Vol Term Structure")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng   = np.random.default_rng(42)
    n     = 700
    dates = pd.date_range("2021-01-01", periods=n, freq="B")

    # Simulate mean-reverting ATM vol (energy options)
    vol_series = [0.45]
    for _ in range(n - 1):
        dv = 0.15 * (0.45 - vol_series[-1]) + 0.02 * rng.normal()
        vol_series.append(max(0.10, vol_series[-1] + dv))
    atm_vol = pd.Series(vol_series, index=dates, name="atm_vol")

    forward = pd.Series(
        np.clip(80 + np.cumsum(rng.normal(0, 1.5, n)), 20, 300), index=dates
    )
    realised = atm_vol * (0.85 + rng.normal(0, 0.15, n))
    realised = pd.Series(np.clip(realised, 0.05, 2.0), index=dates)

    # ATM vol strategy
    cfg   = VolStrategyConfig(lookback=30, entry_zscore=1.5)
    strat = ATMVolStrategy(atm_vol_ts=atm_vol, forward_ts=forward,
                           realised_vol_ts=realised, config=cfg)
    strat.run()
    stats = strat.summary()
    print("\n=== ATM Vol Strategy — Summary ===")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    # Vol surface
    surface = build_synthetic_surface(forward=80.0, atm_vol=0.45, skew=0.06)
    print(f"\nATM vol term structure:\n{surface.term_structure().to_string()}")
    print(f"Skew (M1): {surface.skew(0):.4f}")

    fig1 = strat.plot()
    fig1.savefig("vol_surface_strategy.png", dpi=150, bbox_inches="tight")
    fig2 = plot_vol_surface(surface)
    fig2.savefig("vol_surface_plot.png", dpi=150, bbox_inches="tight")
    print("\nCharts saved.")
