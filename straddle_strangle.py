"""
Straddle & Strangle — Energy Options Vol Strategies
=====================================================
Implements long/short straddle and strangle strategies for
power and gas options markets.

Strategy definitions:

    STRADDLE (ATM):
        Long:  Buy ATM call + Buy ATM put
               → Profits if price moves significantly in either direction
               → Pays time decay (theta), benefits from vol expansion
        Short: Sell ATM call + Sell ATM put
               → Profits if price stays near current level
               → Earns time decay, suffers from large moves

    STRANGLE (OTM):
        Long:  Buy OTM call (K > F) + Buy OTM put (K < F)
               → Cheaper than straddle, needs larger move to profit
               → Lower theta, lower premium at risk
        Short: Sell OTM call + Sell OTM put
               → Cheaper to enter, profits if price stays in range
               → Unlimited risk on both wings (in theory)

    Key Greeks:
        Delta:  ≈ 0 at inception (delta-neutral)
        Gamma:  Positive for longs (accelerating P&L), negative for shorts
        Vega:   Positive for longs (benefits from vol rise)
        Theta:  Negative for longs (time decay), positive for shorts

    Energy market context:
        - High vega sensitivity: energy vol can double overnight (cold snap)
        - Seasonal vol structure: short winter straddles earn theta but carry spike risk
        - Vol of vol: energy vols are themselves highly volatile
        - Realised vs implied: typical vol risk premium of 3-8% in energy

Trading signals:
    1. Realised vol forecast vs current implied vol
    2. Vol regime classification (high/low/trending)
    3. Upcoming events (weather forecasts, inventory releases)
    4. Term structure slope (carry trade on vol)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Literal
from scipy.stats import norm


# ─── Black-76 pricing engine (reuse from vol_surface_model) ──────────────────

def black76(F, K, T, r, sigma, flag="call"):
    if T <= 0 or sigma <= 0:
        return max(0.0, (F - K) if flag == "call" else (K - F))
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df = np.exp(-r * T)
    if flag == "call":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_greeks(F, K, T, r, sigma, flag="call"):
    """Return dict of Greeks for a Black-76 option."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}
    d1  = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2  = d1 - sigma * np.sqrt(T)
    df  = np.exp(-r * T)
    npd = norm.pdf(d1)

    gamma = df * npd / (F * sigma * np.sqrt(T))
    vega  = F * df * npd * np.sqrt(T)
    theta = -(F * df * npd * sigma) / (2 * np.sqrt(T)) / 365

    if flag == "call":
        delta = df * norm.cdf(d1)
    else:
        delta = -df * norm.cdf(-d1)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


# ─── Straddle / Strangle structures ──────────────────────────────────────────

@dataclass
class OptionLeg:
    """Single option leg."""
    flag: Literal["call", "put"]
    strike: float
    position: Literal["long", "short"]  # +1 or -1
    quantity: float = 1.0               # number of contracts


@dataclass
class VolStructure:
    """A multi-leg options structure."""
    name: str
    legs: list
    forward: float
    T: float                            # time to expiry [years]
    implied_vol: float
    rate: float = 0.03

    def premium(self) -> float:
        """Total net premium paid (+) or received (-) [€/MWh]."""
        total = 0.0
        for leg in self.legs:
            sign  = 1.0 if leg.position == "long" else -1.0
            price = black76(self.forward, leg.strike, self.T,
                            self.rate, self.implied_vol, leg.flag)
            total += sign * leg.quantity * price
        return total

    def net_greeks(self) -> dict:
        """Aggregate Greeks across all legs."""
        agg = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for leg in self.legs:
            sign = 1.0 if leg.position == "long" else -1.0
            g    = black76_greeks(self.forward, leg.strike, self.T,
                                  self.rate, self.implied_vol, leg.flag)
            for k in agg:
                agg[k] += sign * leg.quantity * g[k]
        return {k: round(v, 6) for k, v in agg.items()}

    def pnl_at_expiry(self, spot_range: np.ndarray) -> np.ndarray:
        """P&L at expiry across a range of spot prices."""
        prem  = self.premium()
        total = np.zeros_like(spot_range, dtype=float)
        for leg in self.legs:
            sign = 1.0 if leg.position == "long" else -1.0
            if leg.flag == "call":
                payoff = np.maximum(spot_range - leg.strike, 0)
            else:
                payoff = np.maximum(leg.strike - spot_range, 0)
            total += sign * leg.quantity * payoff
        # Net of premium
        net_sign = -1.0 if sum(1 for l in self.legs if l.position == "long") > 0 else 1.0
        return total - prem


def build_straddle(
    forward: float,
    T: float,
    implied_vol: float,
    position: Literal["long", "short"] = "long",
    rate: float = 0.03,
) -> VolStructure:
    """ATM Straddle: call + put at the same strike (= forward)."""
    opp = "short" if position == "long" else "long"
    # For short straddle: sell both legs
    legs = [
        OptionLeg(flag="call", strike=forward, position=position),
        OptionLeg(flag="put",  strike=forward, position=position),
    ]
    return VolStructure(
        name=f"{position.capitalize()} Straddle",
        legs=legs, forward=forward, T=T,
        implied_vol=implied_vol, rate=rate,
    )


def build_strangle(
    forward: float,
    T: float,
    implied_vol: float,
    call_strike_pct: float = 1.10,    # 10% OTM call
    put_strike_pct: float  = 0.90,    # 10% OTM put
    position: Literal["long", "short"] = "long",
    rate: float = 0.03,
) -> VolStructure:
    """OTM Strangle: OTM call + OTM put."""
    legs = [
        OptionLeg(flag="call", strike=forward * call_strike_pct, position=position),
        OptionLeg(flag="put",  strike=forward * put_strike_pct,  position=position),
    ]
    return VolStructure(
        name=f"{position.capitalize()} Strangle ({call_strike_pct:.0%}/{put_strike_pct:.0%})",
        legs=legs, forward=forward, T=T,
        implied_vol=implied_vol, rate=rate,
    )


# ─── Trading strategy ─────────────────────────────────────────────────────────

@dataclass
class StraddleConfig:
    """Strategy configuration."""
    lookback: int              = 30
    entry_zscore: float        = 1.5
    exit_zscore: float         = 0.4
    stop_zscore: float         = 3.5
    structure: str             = "straddle"   # "straddle" or "strangle"
    call_strike_pct: float     = 1.10
    put_strike_pct: float      = 0.90
    T: float                   = 0.25         # option tenor [years]
    rate: float                = 0.03


class StraddleStrangleStrategy:
    """
    Systematic straddle/strangle trading strategy.

    Signal: realised vol vs implied vol
        Realised > Implied → buy vol (long straddle/strangle)
        Realised < Implied → sell vol (short straddle/strangle)

    P&L approximation (delta-neutral, no rebalancing):
        Daily P&L ≈ 0.5 * Gamma * F² * (σ_realised² - σ_implied²) * dt

    Parameters
    ----------
    forward_ts     : pd.Series   Forward price [€/MWh].
    implied_vol_ts : pd.Series   ATM implied vol [decimal].
    realised_vol_ts: pd.Series   Rolling realised vol [decimal].
    config         : StraddleConfig
    """

    def __init__(
        self,
        forward_ts: pd.Series,
        implied_vol_ts: pd.Series,
        realised_vol_ts: pd.Series,
        config: Optional[StraddleConfig] = None,
    ):
        self.fwd      = forward_ts.rename("forward")
        self.iv       = implied_vol_ts.rename("implied_vol")
        self.rv       = realised_vol_ts.rename("realised_vol")
        self.cfg      = config or StraddleConfig()
        self.results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Vol metrics
    # ------------------------------------------------------------------

    def vol_risk_premium(self) -> pd.Series:
        """IV - RV: positive = implied expensive vs realised."""
        vrp = self.iv - self.rv
        vrp.name = "vol_risk_premium"
        return vrp

    def vrp_zscore(self) -> pd.Series:
        """Z-score of VRP for signal generation."""
        vrp    = self.vol_risk_premium()
        mean   = vrp.rolling(self.cfg.lookback).mean()
        std    = vrp.rolling(self.cfg.lookback).std().replace(0, np.nan)
        zscore = (vrp - mean) / std
        zscore.name = "vrp_zscore"
        return zscore

    # ------------------------------------------------------------------
    # Greek-based P&L approximation
    # ------------------------------------------------------------------

    def daily_gamma_pnl(self, position: pd.Series) -> pd.Series:
        """
        Approximate daily P&L from gamma exposure.
        P&L = position * 0.5 * Gamma * F² * (σ_R² - σ_I²) / 252
        """
        pnl = pd.Series(0.0, index=position.index)
        for i in range(1, len(position)):
            pos = position.iloc[i - 1]
            if pos == 0:
                continue
            F    = float(self.fwd.iloc[i])
            iv   = float(self.iv.iloc[i])
            rv   = float(self.rv.iloc[i])
            if np.isnan(F) or np.isnan(iv) or np.isnan(rv):
                continue
            # Gamma of ATM straddle (2 * call gamma)
            T    = max(self.cfg.T, 1/252)
            d1   = 0.5 * iv * np.sqrt(T)
            g    = 2 * np.exp(-self.cfg.rate * T) * norm.pdf(d1) / (F * iv * np.sqrt(T))
            pnl.iloc[i] = pos * 0.5 * g * F**2 * (rv**2 - iv**2) / 252
        pnl.name = "daily_pnl"
        return pnl

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Generate straddle/strangle trading signals and P&L."""
        vrp_z  = self.vrp_zscore()
        vrp    = self.vol_risk_premium()

        position = pd.Series(0.0, index=self.fwd.index)
        current  = 0

        for i in range(self.cfg.lookback, len(vrp_z)):
            z = vrp_z.iloc[i]
            if np.isnan(z): continue
            if current == 0:
                if z > self.cfg.entry_zscore:
                    current = -1    # IV >> RV → sell vol
                elif z < -self.cfg.entry_zscore:
                    current =  1    # IV << RV → buy vol
            elif current == -1:
                if z <= self.cfg.exit_zscore or z <= -self.cfg.stop_zscore:
                    current = 0
            elif current == 1:
                if z >= -self.cfg.exit_zscore or z >= self.cfg.stop_zscore:
                    current = 0
            position.iloc[i] = current

        daily_pnl = self.daily_gamma_pnl(position)

        self.results = pd.DataFrame({
            "forward":      self.fwd,
            "implied_vol":  self.iv,
            "realised_vol": self.rv,
            "vrp":          vrp,
            "vrp_zscore":   vrp_z,
            "position":     position,
            "daily_pnl":    daily_pnl,
            "cum_pnl":      daily_pnl.cumsum(),
        })
        return self.results

    # ------------------------------------------------------------------
    # Structure analysis
    # ------------------------------------------------------------------

    def analyse_structure(
        self,
        F: float, iv: float,
        structure: str = "straddle",
    ) -> dict:
        """Analyse a specific structure at given market conditions."""
        T = self.cfg.T
        r = self.cfg.rate

        if structure == "straddle":
            s = build_straddle(F, T, iv, "long", r)
        else:
            s = build_strangle(F, T, iv, self.cfg.call_strike_pct,
                               self.cfg.put_strike_pct, "long", r)

        greeks  = s.net_greeks()
        premium = s.premium()
        breakeven_up   = F + premium
        breakeven_down = F - premium

        return {
            "structure":        s.name,
            "forward":          F,
            "implied_vol_pct":  round(iv * 100, 2),
            "net_premium":      round(premium, 4),
            "breakeven_up":     round(breakeven_up, 2),
            "breakeven_down":   round(breakeven_down, 2),
            "breakeven_move_pct": round(premium / F * 100, 2),
            **greeks,
        }

    # ------------------------------------------------------------------
    # Summary & plot
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if self.results is None: raise RuntimeError("Call run() first.")
        pnl    = self.results["daily_pnl"].dropna()
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else 0.0
        dd     = (pnl.cumsum() - pnl.cumsum().cummax()).min()
        avg_iv = self.iv.mean()
        avg_rv = self.rv.mean()
        return {
            "structure":        self.cfg.structure,
            "avg_implied_vol":  round(avg_iv * 100, 2),
            "avg_realised_vol": round(avg_rv * 100, 2),
            "avg_vrp_pp":       round((avg_iv - avg_rv) * 100, 2),
            "total_pnl":        round(pnl.sum(), 6),
            "sharpe_ratio":     round(sharpe, 3),
            "max_drawdown":     round(dd, 6),
            "win_rate":         round((pnl > 0).mean(), 3),
            "n_long_vol":       int((self.results["position"] == 1).sum()),
            "n_short_vol":      int((self.results["position"] == -1).sum()),
        }

    def plot(self, figsize=(14, 13)) -> plt.Figure:
        if self.results is None: raise RuntimeError("Call run() first.")
        df  = self.results
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle(f"Straddle/Strangle Vol Strategy — {self.cfg.structure.capitalize()}",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(df.index, df["implied_vol"]  * 100, label="Implied Vol (%)",  color="#c62828", lw=1.0)
        ax.plot(df.index, df["realised_vol"] * 100, label="Realised Vol (%)", color="#1565c0", lw=1.0, ls="--")
        ax.legend(fontsize=8); ax.set_ylabel("Vol (%)"); ax.set_title("Implied vs Realised Vol", fontsize=10)

        ax = axes[1]
        ax.fill_between(df.index, df["vrp"] * 100, 0,
                        where=df["vrp"] > 0, color="red",   alpha=0.4, label="Sell vol")
        ax.fill_between(df.index, df["vrp"] * 100, 0,
                        where=df["vrp"] < 0, color="green", alpha=0.4, label="Buy vol")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=8); ax.set_ylabel("VRP (pp)"); ax.set_title("Vol Risk Premium", fontsize=10)

        ax = axes[2]
        ax.plot(df.index, df["vrp_zscore"], color="#555", lw=0.8)
        ax.axhline(self.cfg.entry_zscore,  color="red",   lw=0.8, ls="--", label="Sell vol threshold")
        ax.axhline(-self.cfg.entry_zscore, color="green", lw=0.8, ls="--", label="Buy vol threshold")
        ax.axhline(0, color="black", lw=0.3)
        sells = df[df["position"] == -1]
        buys  = df[df["position"] ==  1]
        ax.scatter(sells.index, sells["vrp_zscore"], s=14, color="red",   zorder=5)
        ax.scatter(buys.index,  buys["vrp_zscore"],  s=14, color="green", zorder=5)
        ax.legend(fontsize=8); ax.set_ylabel("Z-score"); ax.set_title("VRP Z-score & Signals", fontsize=10)

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


# ─── Payoff diagram utility ───────────────────────────────────────────────────

def plot_payoff_diagram(
    forward: float = 80.0,
    implied_vol: float = 0.40,
    T: float = 0.25,
    figsize=(12, 5),
) -> plt.Figure:
    """Compare long straddle vs long strangle payoff at expiry."""
    spot_range = np.linspace(forward * 0.5, forward * 1.5, 300)

    straddle = build_straddle(forward, T, implied_vol, "long")
    strangle = build_strangle(forward, T, implied_vol, 1.10, 0.90, "long")

    pnl_strad = straddle.pnl_at_expiry(spot_range)
    pnl_stran = strangle.pnl_at_expiry(spot_range)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(spot_range, pnl_strad, label=f"Long Straddle (prem={straddle.premium():.2f})",
            color="#c62828", lw=1.5)
    ax.plot(spot_range, pnl_stran, label=f"Long Strangle (prem={strangle.premium():.2f})",
            color="#1565c0", lw=1.5, ls="--")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(forward, color="#888", lw=0.8, ls=":", label=f"Forward={forward:.1f}")
    ax.fill_between(spot_range, pnl_strad, 0,
                    where=pnl_strad > 0, color="green", alpha=0.12)
    ax.set_xlabel("Spot Price at Expiry [€/MWh]")
    ax.set_ylabel("P&L [€/MWh]")
    ax.set_title(f"Straddle vs Strangle Payoff  |  F={forward}  σ={implied_vol:.0%}  T={T:.2f}Y",
                 fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng   = np.random.default_rng(42)
    n     = 700
    dates = pd.date_range("2021-01-01", periods=n, freq="B")

    # Simulate IV / RV pair with vol risk premium
    iv_series = [0.40]
    for _ in range(n - 1):
        iv_series.append(max(0.10, iv_series[-1] + 0.12*(0.40 - iv_series[-1]) + 0.015*rng.normal()))
    iv = pd.Series(iv_series, index=dates)
    rv = pd.Series(np.clip(np.array(iv_series)*0.88 + rng.normal(0, 0.03, n), 0.08, 1.5), index=dates)
    fwd= pd.Series(np.clip(80 + np.cumsum(rng.normal(0, 1.2, n)), 20, 300), index=dates)

    cfg   = StraddleConfig(lookback=30, entry_zscore=1.5, structure="straddle", T=0.25)
    strat = StraddleStrangleStrategy(forward_ts=fwd, implied_vol_ts=iv,
                                     realised_vol_ts=rv, config=cfg)
    strat.run()
    stats = strat.summary()
    print("\n=== Straddle/Strangle Strategy — Summary ===")
    for k, v in stats.items():
        print(f"  {k:28s}: {v}")

    analysis = strat.analyse_structure(F=80.0, iv=0.40, structure="straddle")
    print("\n  Structure Analysis (Long Straddle @ F=80, σ=40%):")
    for k, v in analysis.items():
        print(f"    {k:25s}: {v}")

    fig1 = strat.plot()
    fig1.savefig("straddle_strategy.png", dpi=150, bbox_inches="tight")
    fig2 = plot_payoff_diagram(forward=80.0, implied_vol=0.40, T=0.25)
    fig2.savefig("straddle_payoff.png", dpi=150, bbox_inches="tight")
    print("\nCharts saved.")
