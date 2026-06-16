"""
Optimal Execution — Almgren-Chriss Framework (Energy Adaptation)
=================================================================
Implements the Almgren-Chriss (2001) optimal liquidation model
adapted for European power and gas trading.

Theory:
    The Almgren-Chriss model finds the optimal execution trajectory
    that minimises a combination of:
        1. Expected cost (market impact)
        2. Execution risk (variance of cost, controlled by risk aversion λ)

    The optimal strategy balances:
        - Trading FAST → less risk but more market impact
        - Trading SLOW → less impact but more price risk (inventory risk)

    Model setup:
        X₀ = initial position to liquidate [MW or MWh]
        T  = execution horizon [hours or days]
        N  = number of trading intervals
        τ  = T/N (interval length)

    Price impact:
        Temporary impact:  h(v) = η * v    [linear, recovers after trade]
        Permanent impact:  g(v) = γ * v    [linear, permanent price depression]

    where v = trading rate [MW/hour]

    Optimal trajectory (closed form):
        x(t) = X₀ * sinh(κ(T-t)) / sinh(κT)

        where κ = sqrt(λ σ² / η)
            - Large λ (risk-averse) → fast execution (large κ)
            - Small λ (risk-neutral) → slow execution (TWAP-like)

    Expected cost:
        E[C] = γ X₀²/2 + η X₀² * (κT coth(κT) - 1) / (κT)

    Variance:
        Var[C] = σ² η X₀² * (2κT coth(2κT) - κT coth(κT)² - 1) / (2κ³T)

    Energy market adaptations:
        1. Time-varying liquidity: η(t) changes throughout the day
        2. Session structure: discrete auction sessions → discrete schedule
        3. Shape constraints: energy products have delivery profiles
        4. Cross-product optimisation: simultaneous power + gas + EUA
        5. Risk measured in MWh-days (exposure to price changes)

References:
    Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio Transactions."
    Journal of Risk, 3(2), 5-39.
    
    Cartea, Á. & Jaimungal, S. (2015). "Optimal Execution with Limit and Market Orders."
    Quantitative Finance, 15(8), 1279-1291.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from scipy.optimize import minimize


# ─── Market impact parameters ─────────────────────────────────────────────────

@dataclass
class MarketImpactParams:
    """
    Market impact parameters for an energy product.

    Parameters
    ----------
    sigma       : float   Daily price volatility [€/MWh]
    eta         : float   Temporary impact coefficient [€/(MWh²)]
    gamma       : float   Permanent impact coefficient [€/(MWh²)]
    adv         : float   Average Daily Volume [MWh] (for normalisation)
    bid_ask     : float   Half bid-ask spread [€/MWh]
    market      : str     Market name
    product     : str     Product description
    """
    sigma:   float  = 3.0           # daily vol [€/MWh]
    eta:     float  = 0.002         # temporary impact [€/MWh per MW traded]
    gamma:   float  = 0.001         # permanent impact [€/MWh per MW traded]
    adv:     float  = 5000.0        # average daily volume [MWh]
    bid_ask: float  = 0.25          # half spread [€/MWh]
    market:  str    = "EPEX_DE"
    product: str    = "Power_DE_Continuous"

    def kyle_lambda(self) -> float:
        """Kyle's lambda: price impact per unit volume."""
        return self.gamma + self.eta


# ─── Almgren-Chriss optimal trajectory ───────────────────────────────────────

class AlmgrenChriss:
    """
    Almgren-Chriss optimal execution model.

    Computes the optimal trading schedule that minimises
    risk-adjusted execution cost.

    Parameters
    ----------
    X0          : float   Initial position to execute [MW]
    T           : float   Execution horizon [hours]
    N           : int     Number of trading intervals
    lambda_     : float   Risk aversion parameter (higher = faster execution)
    impact      : MarketImpactParams
    """

    def __init__(
        self,
        X0: float,
        T: float,
        N: int,
        lambda_: float,
        impact: Optional[MarketImpactParams] = None,
    ):
        self.X0      = X0
        self.T       = T
        self.N       = N
        self.lambda_ = lambda_
        self.impact  = impact or MarketImpactParams()
        self.tau     = T / N             # interval length [hours]

    # ------------------------------------------------------------------
    # Optimal trajectory
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> float:
        """
        Mean reversion-like parameter of optimal strategy.
        κ = sqrt(λ σ² / η)
        Large κ → aggressive execution (risk-averse).
        """
        lam   = self.lambda_
        sigma = self.impact.sigma / np.sqrt(252 * 6.5)  # per hour
        eta   = self.impact.eta
        if eta <= 0:
            return 0.0
        return np.sqrt(lam * sigma**2 / eta)

    def optimal_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optimal inventory trajectory x(t) and trading rates v(t).

        Returns
        -------
        times      : np.ndarray   Time points [0, τ, 2τ, ..., T]
        inventory  : np.ndarray   Remaining position at each time point
        """
        k    = self.kappa
        X0   = self.X0
        T    = self.T
        N    = self.N
        tau  = self.tau
        times = np.linspace(0, T, N + 1)

        if k * T < 1e-6 or np.sinh(k * T) < 1e-10:
            # Near risk-neutral: TWAP
            inventory = X0 * (1 - times / T)
        else:
            inventory = X0 * np.sinh(k * (T - times)) / np.sinh(k * T)

        return times, inventory

    def trading_rates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return trading times and rates v(t) = -dx/dt [MW/hour].
        Positive = selling, negative = buying.
        """
        times, inv = self.optimal_trajectory()
        rates = -np.diff(inv) / self.tau
        return times[:-1], rates

    # ------------------------------------------------------------------
    # Cost analysis
    # ------------------------------------------------------------------

    def expected_cost(self) -> float:
        """
        Expected total execution cost [€].
        E[C] = γ X₀²/2 + η * f(κ, T)
        """
        k    = self.kappa
        X0   = self.X0
        eta  = self.impact.eta
        gam  = self.impact.gamma
        T    = self.T

        permanent_cost = gam * X0**2 / 2

        if k * T < 1e-6:
            # Risk-neutral limit: TWAP cost
            temporary_cost = eta * X0**2 / T
        else:
            coth_kT = 1 / np.tanh(k * T)
            temporary_cost = eta * X0**2 * (k * T * coth_kT - 1) / (k * T)

        return permanent_cost + temporary_cost

    def execution_variance(self) -> float:
        """Variance of execution cost [€²]."""
        k     = self.kappa
        X0    = self.X0
        sigma = self.impact.sigma / np.sqrt(252 * 6.5)
        eta   = self.impact.eta
        T     = self.T

        if k * T < 1e-6:
            return sigma**2 * eta * X0**2 * T / 3

        coth_kT  = 1 / np.tanh(k * T)
        coth_2kT = 1 / np.tanh(2 * k * T)
        var = (sigma**2 * eta * X0**2 *
               (2 * k * T * coth_2kT - (k * T * coth_kT)**2 - 1) /
               (2 * k**3 * T))
        return max(0.0, var)

    def efficient_frontier(
        self,
        n_lambdas: int = 50,
        lambda_range: Tuple[float, float] = (1e-6, 1.0),
    ) -> pd.DataFrame:
        """
        Compute the efficient frontier of (expected cost, std dev of cost)
        for a range of risk aversion parameters.
        """
        lambdas = np.logspace(
            np.log10(lambda_range[0]),
            np.log10(lambda_range[1]),
            n_lambdas,
        )
        records = []
        for lam in lambdas:
            ac = AlmgrenChriss(self.X0, self.T, self.N, lam, self.impact)
            records.append({
                "lambda":         lam,
                "expected_cost":  ac.expected_cost(),
                "cost_std":       np.sqrt(max(0, ac.execution_variance())),
                "kappa":          ac.kappa,
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Comparison with TWAP
    # ------------------------------------------------------------------

    def twap_cost(self) -> float:
        """Cost of TWAP benchmark (λ → 0 limit)."""
        ac_twap = AlmgrenChriss(self.X0, self.T, self.N, 1e-8, self.impact)
        return ac_twap.expected_cost()

    def cost_vs_twap(self) -> float:
        """Cost saving vs TWAP [€]."""
        return self.twap_cost() - self.expected_cost()

    # ------------------------------------------------------------------
    # Schedule as DataFrame
    # ------------------------------------------------------------------

    def schedule(self, start_time: pd.Timestamp) -> pd.DataFrame:
        """
        Return complete execution schedule as a DataFrame.
        """
        times, inv  = self.optimal_trajectory()
        _, rates    = self.trading_rates()

        abs_times = [start_time + pd.Timedelta(hours=t) for t in times]

        df = pd.DataFrame({
            "time":         abs_times,
            "inventory_mw": inv,
        })

        rate_times  = abs_times[:-1]
        rate_df     = pd.DataFrame({
            "time":         rate_times,
            "trade_rate_mw_h":   rates,
            "trade_volume_mw":   rates * self.tau,
        })

        return df.set_index("time"), rate_df.set_index("time")


# ─── Time-varying liquidity extension ────────────────────────────────────────

class TVLAlmgrenChriss:
    """
    Almgren-Chriss with time-varying liquidity (TVL extension).

    In energy markets, market liquidity varies dramatically throughout the day:
    - Peak liquidity: around DA publication, market open
    - Low liquidity:  early morning, late evening, weekends

    The TVL model uses a time-varying η(t) to schedule more aggressively
    during liquid periods and passively during thin markets.

    Parameters
    ----------
    X0              : float
    start_time      : pd.Timestamp
    end_time        : pd.Timestamp
    impact          : MarketImpactParams
    lambda_         : float
    liquidity_profile: pd.Series  η(t) relative to base (1.0 = normal liquidity)
    """

    def __init__(
        self,
        X0: float,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        impact: Optional[MarketImpactParams] = None,
        lambda_: float = 0.01,
        liquidity_profile: Optional[pd.Series] = None,
    ):
        self.X0          = X0
        self.start        = start_time
        self.end          = end_time
        self.impact       = impact or MarketImpactParams()
        self.lambda_      = lambda_
        self.liq_profile  = liquidity_profile

    def _get_liquidity_weights(self, times: pd.DatetimeIndex) -> np.ndarray:
        """Get inverse liquidity weights (more liquid = lower η = trade more)."""
        if self.liq_profile is None:
            return np.ones(len(times))
        weights = np.array([
            float(self.liq_profile.get(t, 1.0)) for t in times
        ])
        return np.maximum(weights, 0.1)

    def optimal_schedule(self, N: int = 20) -> pd.DataFrame:
        """
        Compute optimal schedule with time-varying liquidity via numerical optimisation.
        """
        times    = pd.date_range(self.start, self.end, periods=N + 1)
        T        = (self.end - self.start).total_seconds() / 3600
        tau      = T / N
        sigma_h  = self.impact.sigma / np.sqrt(252 * 6.5)
        eta_base = self.impact.eta
        gam      = self.impact.gamma
        lam      = self.lambda_

        liq_weights = self._get_liquidity_weights(times[:-1])
        eta_t       = eta_base / liq_weights   # lower η where more liquid

        def total_utility(trades: np.ndarray) -> float:
            """Objective: E[cost] + λ * Var[cost]."""
            if trades.sum() <= 0:
                return 1e10
            inv    = np.concatenate([[self.X0], self.X0 - np.cumsum(trades)])
            rates  = trades / tau

            # Temporary impact cost
            temp_cost = np.sum(eta_t * rates**2 * tau)
            # Permanent impact cost
            perm_cost = gam * np.sum(rates * inv[:-1]) * tau
            # Execution variance (simplified)
            exec_var  = sigma_h**2 * np.sum(inv[1:]**2) * tau

            return temp_cost + perm_cost + lam * exec_var

        # Constraints: sum of trades = X0, each trade >= 0
        from scipy.optimize import LinearConstraint, Bounds
        constraints = [{"type": "eq", "fun": lambda x: x.sum() - self.X0}]
        bounds      = [(0, self.X0)] * N
        x0          = np.full(N, self.X0 / N)

        try:
            res = minimize(total_utility, x0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"maxiter": 500, "ftol": 1e-8})
            trades = res.x
        except Exception:
            trades = np.full(N, self.X0 / N)

        inventory = np.concatenate([[self.X0], self.X0 - np.cumsum(trades)])

        return pd.DataFrame({
            "time":            times,
            "inventory_mw":    inventory,
            "trade_volume_mw": np.append(trades, 0),
            "eta_t":           np.append(eta_t, eta_t[-1]),
            "liq_weight":      np.append(liq_weights, liq_weights[-1]),
        }).set_index("time")


# ─── Comparison and plotting ──────────────────────────────────────────────────

def plot_ac_analysis(
    X0: float = 100.0,
    T: float  = 6.0,
    N: int    = 20,
    impact: Optional[MarketImpactParams] = None,
    figsize=(14, 12),
) -> plt.Figure:
    """Full Almgren-Chriss analysis plot."""
    if impact is None:
        impact = MarketImpactParams()

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f"Almgren-Chriss Optimal Execution  |  X₀={X0:.0f} MW  T={T:.0f}h",
                 fontsize=13, fontweight="bold")

    lambdas_to_show = [1e-4, 0.005, 0.05, 0.5]
    colours         = ["#1565c0", "#2e7d32", "#e65100", "#c62828"]
    labels          = ["λ=1e-4 (passive)", "λ=0.005", "λ=0.05", "λ=0.5 (aggressive)"]
    start_t         = pd.Timestamp("2024-01-15 08:00")

    # Panel 1: Trajectory comparison
    ax = axes[0, 0]
    for lam, col, lab in zip(lambdas_to_show, colours, labels):
        ac = AlmgrenChriss(X0, T, N, lam, impact)
        times, inv = ac.optimal_trajectory()
        ax.plot(times, inv, color=col, lw=1.5, label=lab)
    ax.set_xlabel("Time [hours]"); ax.set_ylabel("Inventory [MW]")
    ax.set_title("Optimal Inventory Trajectory", fontsize=10)
    ax.legend(fontsize=7)

    # Panel 2: Trading rates
    ax = axes[0, 1]
    for lam, col, lab in zip(lambdas_to_show, colours, labels):
        ac = AlmgrenChriss(X0, T, N, lam, impact)
        t_rates, rates = ac.trading_rates()
        ax.plot(t_rates, rates, color=col, lw=1.5, label=lab)
    ax.set_xlabel("Time [hours]"); ax.set_ylabel("Trade Rate [MW/h]")
    ax.set_title("Trading Rates", fontsize=10)
    ax.legend(fontsize=7)

    # Panel 3: Efficient frontier
    ax = axes[1, 0]
    ac_base = AlmgrenChriss(X0, T, N, 0.01, impact)
    ef = ac_base.efficient_frontier(n_lambdas=60, lambda_range=(1e-5, 2.0))
    ax.plot(ef["cost_std"], ef["expected_cost"], color="#333", lw=2)
    ax.set_xlabel("Std Dev of Cost [€]"); ax.set_ylabel("Expected Cost [€]")
    ax.set_title("Efficient Frontier (Cost vs Risk)", fontsize=10)
    # Mark specific lambdas
    for lam, col, lab in zip(lambdas_to_show, colours, labels):
        ac  = AlmgrenChriss(X0, T, N, lam, impact)
        ax.scatter(np.sqrt(max(0,ac.execution_variance())), ac.expected_cost(),
                   color=col, s=60, zorder=5, label=lab)
    ax.legend(fontsize=7)

    # Panel 4: Cost breakdown
    ax = axes[1, 1]
    lam_grid = np.logspace(-5, 0, 50)
    ec       = [AlmgrenChriss(X0, T, N, l, impact).expected_cost() for l in lam_grid]
    ax.semilogx(lam_grid, ec, color="#1565c0", lw=2)
    ax.axhline(AlmgrenChriss(X0, T, N, 1e-8, impact).expected_cost(),
               color="red", lw=1, ls="--", label="TWAP cost")
    ax.set_xlabel("Risk Aversion λ"); ax.set_ylabel("Expected Cost [€]")
    ax.set_title("Expected Cost vs Risk Aversion", fontsize=10)
    ax.legend(fontsize=8)

    # Panel 5: Schedule at λ=0.05
    ax = axes[2, 0]
    ac = AlmgrenChriss(X0, T, N, 0.05, impact)
    inv_df, rate_df = ac.schedule(start_t)
    ax.bar(range(len(rate_df)), rate_df["trade_volume_mw"].values,
           color="#2e7d32", alpha=0.7)
    ax.set_ylabel("Trade Volume [MW]"); ax.set_xlabel("Interval")
    ax.set_title("Trade Schedule (λ=0.05)", fontsize=10)

    # Panel 6: Kappa vs lambda
    ax = axes[2, 1]
    kappas = [AlmgrenChriss(X0, T, N, l, impact).kappa for l in lam_grid]
    ax.loglog(lam_grid, kappas, color="#e65100", lw=2)
    ax.set_xlabel("Risk Aversion λ"); ax.set_ylabel("κ (execution urgency)")
    ax.set_title("Execution Urgency κ vs Risk Aversion", fontsize=10)

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    impact = MarketImpactParams(
        sigma   = 4.0,       # €/MWh daily vol
        eta     = 0.003,     # temporary impact
        gamma   = 0.001,     # permanent impact
        adv     = 8000.0,    # MWh average daily volume
        market  = "EPEX_DE",
        product = "Power_DE_Continuous",
    )

    X0 = 200.0    # 200 MW to execute
    T  = 6.0      # over 6 hours
    N  = 24       # 24 intervals (15-min each)

    print("=== Almgren-Chriss Optimal Execution ===\n")
    print(f"  Position to execute: {X0:.0f} MW")
    print(f"  Horizon:             {T:.0f} hours ({N} intervals)")
    print(f"  Market:              {impact.market}")

    print("\n  Cost comparison across risk aversion levels:")
    print(f"  {'Lambda':>10}  {'E[Cost] €':>12}  {'Std[Cost] €':>12}  {'vs TWAP €':>12}")
    twap_cost = AlmgrenChriss(X0, T, N, 1e-8, impact).expected_cost()
    for lam in [1e-5, 0.001, 0.01, 0.10, 0.50]:
        ac = AlmgrenChriss(X0, T, N, lam, impact)
        ec = ac.expected_cost()
        sd = np.sqrt(max(0, ac.execution_variance()))
        print(f"  {lam:>10.4f}  {ec:>12.2f}  {sd:>12.2f}  {ec - twap_cost:>12.2f}")

    # Efficient frontier + schedule
    fig = plot_ac_analysis(X0=X0, T=T, N=N, impact=impact)
    fig.savefig("optimal_execution_ac.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → optimal_execution_ac.png")

    # Time-varying liquidity example
    times_liq = pd.date_range("2024-01-15 08:00", "2024-01-15 14:00", freq="15min")
    liq = pd.Series(
        [0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.9,
         0.8, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5,
         0.5][:len(times_liq)],
        index=times_liq, name="liquidity"
    )
    tvl = TVLAlmgrenChriss(
        X0=X0,
        start_time=times_liq[0],
        end_time=times_liq[-1],
        impact=impact,
        lambda_=0.01,
        liquidity_profile=liq,
    )
    tvl_schedule = tvl.optimal_schedule(N=20)
    print(f"\nTVL Schedule (first 5 intervals):")
    print(tvl_schedule[["inventory_mw","trade_volume_mw","liq_weight"]].head().round(3).to_string())
