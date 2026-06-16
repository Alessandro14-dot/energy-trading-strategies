"""
Hidden Markov Model — Energy Market Regime Detection
======================================================
Detects market regimes in energy prices using a Hidden Markov Model (HMM).

Why regime detection matters:
    Energy markets alternate between distinct regimes:
        - LOW VOL / CONTANGO:   Stable prices, forward > spot, abundant supply
        - HIGH VOL / BACKWARDATION: Volatile, spot > forward, tight supply
        - SPIKE REGIME:         Extreme prices, grid stress, scarcity events
        - NEGATIVE PRICE REGIME: Oversupply, excess renewables, mandatory curtailment

    Trading strategy effectiveness varies strongly by regime:
        - Mean reversion works best in Low Vol
        - Momentum works best in High Vol / Trending
        - Straddles make money in Spike regime
        - Storage injection is optimal in Contango regime

    Regime-aware strategies:
        → Select strategies based on current regime
        → Adjust position sizing and risk limits per regime
        → Avoid mean-reversion in trending/spike regimes

Hidden Markov Model:
    State space:    Z(t) ∈ {1, 2, ..., K}     (hidden regimes)
    Observations:   Y(t) = (returns, vol, spread, ...)

    Model parameters:
        π       Initial state probabilities
        A       Transition matrix A[i,j] = P(Z(t)=j | Z(t-1)=i)
        B       Emission parameters (Gaussian: μ_k, Σ_k for each state k)

    Inference:
        Viterbi algorithm:     Most likely state sequence
        Forward-backward (EM): Parameter estimation (Baum-Welch)
        Smoothed probabilities: P(Z(t)=k | Y(1:T))

    Feature vector for HMM observation:
        - Daily log return
        - Rolling volatility (5-day, 20-day)
        - Spread z-score
        - Volume anomaly
        - Forward curve slope (contango/backwardation)

    Typical energy regimes (K=3 or K=4):
        Regime 0: Low vol, contango, bearish (storage build season)
        Regime 1: Medium vol, transitional
        Regime 2: High vol, backwardation, bullish (winter/scarcity)
        Regime 3: Spike / extreme (optional 4th state)

Note:
    This implementation uses scikit-learn's GaussianMixture as a simpler
    alternative to full HMM (hmmlearn library).
    For production HMM, install: pip install hmmlearn
    and use hmmlearn.hmm.GaussianHMM.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn import hmm as hmmlearn_hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False


# ─── Regime feature engineering ──────────────────────────────────────────────

class RegimeFeatureEngine:
    """
    Build observation features for HMM regime detection.

    Parameters
    ----------
    prices          : pd.Series   Price time series [€/MWh]
    forward_prices  : pd.Series   Forward/futures price [€/MWh]
    volumes         : pd.Series   Trading volume (optional)
    spread          : pd.Series   Spread series (optional, e.g. spark spread)
    """

    def __init__(
        self,
        prices: pd.Series,
        forward_prices: Optional[pd.Series] = None,
        volumes: Optional[pd.Series]        = None,
        spread: Optional[pd.Series]         = None,
    ):
        self.prices  = prices.rename("price")
        self.forward = forward_prices
        self.volumes = volumes
        self.spread  = spread

    def build_observation_matrix(
        self,
        vol_windows: List[int] = [5, 20],
        zscore_window: int     = 20,
    ) -> pd.DataFrame:
        """
        Build feature matrix for HMM observations.
        Each row is the observation vector for one time step.
        """
        dfs = []

        # Log returns
        log_ret = np.log(self.prices / self.prices.shift(1))
        dfs.append(log_ret.rename("log_return").to_frame())

        # Signed return (direction)
        dfs.append((np.sign(log_ret)).rename("return_sign").to_frame())

        # Rolling volatility (annualised)
        for w in vol_windows:
            rv = log_ret.rolling(w).std() * np.sqrt(252)
            dfs.append(rv.rename(f"vol_{w}d").to_frame())

        # Volatility ratio (short/long vol = vol regime indicator)
        if len(vol_windows) >= 2:
            vr = (log_ret.rolling(vol_windows[0]).std() /
                  log_ret.rolling(vol_windows[-1]).std().replace(0, np.nan))
            dfs.append(vr.rename("vol_ratio").to_frame())

        # Price z-score (level vs rolling mean)
        p_mean = self.prices.rolling(zscore_window).mean()
        p_std  = self.prices.rolling(zscore_window).std().replace(0, np.nan)
        p_z    = (self.prices - p_mean) / p_std
        dfs.append(p_z.rename("price_zscore").to_frame())

        # Absolute return (vol proxy)
        dfs.append(log_ret.abs().rename("abs_return").to_frame())

        # Forward curve slope (contango/backwardation indicator)
        if self.forward is not None:
            slope = (self.forward - self.prices) / self.prices.replace(0, np.nan)
            m  = slope.rolling(zscore_window).mean()
            s  = slope.rolling(zscore_window).std().replace(0, np.nan)
            dfs.append(((slope - m) / s).rename("curve_slope_z").to_frame())

        # Volume anomaly
        if self.volumes is not None:
            vol_ma = self.volumes.rolling(20).mean()
            vol_z  = (self.volumes - vol_ma) / self.volumes.rolling(20).std().replace(0, np.nan)
            dfs.append(vol_z.rename("volume_z").to_frame())

        # Spread features
        if self.spread is not None:
            sp_m = self.spread.rolling(zscore_window).mean()
            sp_s = self.spread.rolling(zscore_window).std().replace(0, np.nan)
            dfs.append(((self.spread - sp_m) / sp_s).rename("spread_z").to_frame())

        obs = pd.concat(dfs, axis=1).dropna()
        return obs


# ─── HMM Regime Detector ─────────────────────────────────────────────────────

@dataclass
class HMMConfig:
    """HMM model configuration."""
    n_regimes: int      = 3           # number of hidden states
    n_iter: int         = 100         # EM iterations
    covariance_type: str= "full"      # "full", "diag", "spherical"
    use_hmmlearn: bool  = True        # use hmmlearn if available
    regime_names: List[str] = field(
        default_factory=lambda: ["Low-Vol/Contango", "Mid-Vol/Transition", "High-Vol/Backwardation"]
    )
    regime_colours: List[str] = field(
        default_factory=lambda: ["#1565c0", "#f9a825", "#c62828"]
    )


class HMMRegimeDetector:
    """
    Hidden Markov Model for energy market regime detection.

    Fits a Gaussian HMM to price observation features and
    assigns each time step to a market regime.

    Parameters
    ----------
    config : HMMConfig
    """

    def __init__(self, config: Optional[HMMConfig] = None):
        self.cfg     = config or HMMConfig()
        self.model   = None
        self.scaler  = StandardScaler()
        self.is_fitted = False
        self.regime_stats: Dict[int, dict] = {}

    def fit(self, observations: pd.DataFrame) -> "HMMRegimeDetector":
        """Fit HMM to observation matrix."""
        X_sc = self.scaler.fit_transform(observations.values)

        if HMMLEARN_AVAILABLE and self.cfg.use_hmmlearn:
            self.model = hmmlearn_hmm.GaussianHMM(
                n_components    = self.cfg.n_regimes,
                covariance_type = self.cfg.covariance_type,
                n_iter          = self.cfg.n_iter,
                random_state    = 42,
            )
            self.model.fit(X_sc)
        else:
            # Fallback: Gaussian Mixture Model (no temporal structure)
            self.model = GaussianMixture(
                n_components    = self.cfg.n_regimes,
                covariance_type = self.cfg.covariance_type,
                n_init          = 5,
                random_state    = 42,
                max_iter        = self.cfg.n_iter,
            )
            self.model.fit(X_sc)

        self.is_fitted = True
        return self

    def predict_regimes(self, observations: pd.DataFrame) -> pd.Series:
        """
        Predict the most likely regime for each time step.
        Returns pd.Series of regime labels {0, 1, ..., K-1}.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        X_sc = self.scaler.transform(observations.values)

        if HMMLEARN_AVAILABLE and isinstance(self.model, hmmlearn_hmm.GaussianHMM):
            states = self.model.predict(X_sc)
        else:
            states = self.model.predict(X_sc)

        regimes = pd.Series(states, index=observations.index, name="regime")
        return self._relabel_regimes(regimes, observations)

    def predict_proba(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Return regime probabilities for each time step."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        X_sc = self.scaler.transform(observations.values)

        if HMMLEARN_AVAILABLE and isinstance(self.model, hmmlearn_hmm.GaussianHMM):
            _, proba = self.model.score_samples(X_sc)
            proba    = np.exp(proba - proba.max(axis=1, keepdims=True))
            proba    = proba / proba.sum(axis=1, keepdims=True)
        else:
            proba = self.model.predict_proba(X_sc)

        cols = [f"p_regime_{k}" for k in range(self.cfg.n_regimes)]
        return pd.DataFrame(proba, index=observations.index, columns=cols)

    def _relabel_regimes(
        self, regimes: pd.Series, observations: pd.DataFrame
    ) -> pd.Series:
        """
        Re-order regime labels so that:
            Regime 0 = lowest volatility
            Regime K-1 = highest volatility
        """
        vol_col = [c for c in observations.columns if "vol_20" in c or "vol_5" in c]
        if not vol_col:
            return regimes
        vol_by_regime = {}
        for k in range(self.cfg.n_regimes):
            mask = regimes == k
            if mask.sum() > 0:
                vol_by_regime[k] = observations.loc[mask, vol_col[0]].mean()
        sorted_regimes = sorted(vol_by_regime, key=vol_by_regime.get)
        mapping = {old: new for new, old in enumerate(sorted_regimes)}
        return regimes.map(mapping)

    # ------------------------------------------------------------------
    # Regime characterisation
    # ------------------------------------------------------------------

    def characterise_regimes(
        self, regimes: pd.Series, prices: pd.Series
    ) -> pd.DataFrame:
        """
        Compute statistics for each detected regime.
        """
        log_ret = np.log(prices / prices.shift(1)).dropna()
        common  = regimes.index.intersection(log_ret.index)
        records = []

        for k in range(self.cfg.n_regimes):
            mask  = regimes.loc[common] == k
            ret_k = log_ret.loc[common][mask]
            px_k  = prices.loc[common][mask]

            records.append({
                "regime":        k,
                "name":          self.cfg.regime_names[k] if k < len(self.cfg.regime_names) else f"Regime {k}",
                "n_days":        int(mask.sum()),
                "pct_of_time":   round(mask.mean() * 100, 1),
                "avg_price":     round(px_k.mean(), 2),
                "avg_return":    round(ret_k.mean() * 100, 4),
                "annualised_vol":round(ret_k.std() * np.sqrt(252) * 100, 2),
                "sharpe":        round((ret_k.mean() / ret_k.std()) * np.sqrt(252), 3) if ret_k.std() > 0 else 0.0,
                "skewness":      round(ret_k.skew(), 3),
                "kurtosis":      round(ret_k.kurtosis(), 3),
                "max_1d_gain":   round(ret_k.max() * 100, 2),
                "max_1d_loss":   round(ret_k.min() * 100, 2),
            })

        return pd.DataFrame(records).set_index("regime")

    # ------------------------------------------------------------------
    # Transition matrix analysis
    # ------------------------------------------------------------------

    def transition_matrix(self, regimes: pd.Series) -> pd.DataFrame:
        """Empirical transition matrix from detected regime sequence."""
        K = self.cfg.n_regimes
        T = np.zeros((K, K))
        vals = regimes.values
        for t in range(len(vals) - 1):
            i, j = int(vals[t]), int(vals[t+1])
            if 0 <= i < K and 0 <= j < K:
                T[i, j] += 1
        row_sums = T.sum(axis=1, keepdims=True)
        T = T / np.maximum(row_sums, 1)
        names = [f"→R{k}" for k in range(K)]
        index = [f"R{k}" for k in range(K)]
        return pd.DataFrame(T, index=index, columns=names).round(3)

    # ------------------------------------------------------------------
    # Regime-conditional strategy allocation
    # ------------------------------------------------------------------

    def strategy_allocation(self, regime: int) -> dict:
        """
        Suggest strategy mix based on detected regime.
        Returns recommended weights for each strategy family.
        """
        allocations = {
            0: {  # Low-Vol / Contango
                "mean_reversion":   0.40,
                "stat_arb":         0.30,
                "storage_injection":0.20,
                "vol_selling":      0.10,
            },
            1: {  # Mid-Vol / Transition
                "mean_reversion":   0.25,
                "momentum":         0.25,
                "fundamental":      0.25,
                "vol_neutral":      0.25,
            },
            2: {  # High-Vol / Backwardation
                "momentum":         0.35,
                "vol_buying":       0.25,
                "fundamental":      0.25,
                "storage_withdrawal":0.15,
            },
        }
        default = {"balanced": 1.0}
        return allocations.get(regime, default)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(
        self,
        regimes: pd.Series,
        prices: pd.Series,
        proba: Optional[pd.DataFrame] = None,
        figsize=(14, 12),
    ) -> plt.Figure:
        """Full regime detection plot."""
        n_rows = 4 if proba is not None else 3
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
        fig.suptitle("HMM Market Regime Detection — Energy Markets",
                     fontsize=13, fontweight="bold")
        colours = self.cfg.regime_colours

        # Panel 1: Price with regime background
        ax = axes[0]
        ax.plot(prices.index, prices.values, color="#333", lw=0.8, zorder=3)
        common = regimes.index.intersection(prices.index)
        for k in range(self.cfg.n_regimes):
            mask = regimes.loc[common] == k
            if mask.any():
                ax.fill_between(common, prices.loc[common].min(), prices.loc[common].max(),
                                where=mask, alpha=0.18, color=colours[k])
        patches = [mpatches.Patch(color=colours[k],
                   label=self.cfg.regime_names[k] if k < len(self.cfg.regime_names) else f"R{k}")
                   for k in range(self.cfg.n_regimes)]
        ax.legend(handles=patches, fontsize=8, loc="upper left")
        ax.set_ylabel("€/MWh"); ax.set_title("Price with Regime Background", fontsize=10)

        # Panel 2: Regime sequence
        ax = axes[1]
        regime_vals = regimes.reindex(prices.index).ffill()
        for k in range(self.cfg.n_regimes):
            mask = regime_vals == k
            ax.fill_between(prices.index, k, k+1,
                            where=mask, color=colours[k], alpha=0.8)
        ax.set_yticks([k + 0.5 for k in range(self.cfg.n_regimes)])
        ax.set_yticklabels(
            [self.cfg.regime_names[k] if k < len(self.cfg.regime_names)
             else f"Regime {k}" for k in range(self.cfg.n_regimes)],
            fontsize=8,
        )
        ax.set_title("Regime Sequence", fontsize=10)

        # Panel 3: Annualised vol per regime
        ax = axes[2]
        log_ret = np.log(prices / prices.shift(1))
        roll_vol = log_ret.rolling(10).std() * np.sqrt(252) * 100
        ax.plot(roll_vol.index, roll_vol.values, color="#555", lw=0.8, label="10d Ann. Vol (%)")
        for k in range(self.cfg.n_regimes):
            mask = regime_vals.reindex(roll_vol.index) == k
            ax.fill_between(roll_vol.index, roll_vol, 0,
                            where=mask, alpha=0.20, color=colours[k])
        ax.legend(fontsize=8); ax.set_ylabel("Vol (%)")
        ax.set_title("Realised Volatility by Regime", fontsize=10)

        # Panel 4: Regime probabilities
        if proba is not None:
            ax = axes[3]
            for k in range(self.cfg.n_regimes):
                col = f"p_regime_{k}"
                if col in proba.columns:
                    lbl = self.cfg.regime_names[k] if k < len(self.cfg.regime_names) else f"R{k}"
                    ax.plot(proba.index, proba[col], color=colours[k], lw=0.8,
                            label=lbl, alpha=0.85)
            ax.legend(fontsize=8); ax.set_ylabel("Probability")
            ax.set_title("Regime Probabilities", fontsize=10)
            ax.set_xlabel("Date")
        else:
            axes[-1].set_xlabel("Date")

        plt.tight_layout()
        return fig


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng   = np.random.default_rng(42)
    n     = 1000
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    doy   = np.array([d.dayofyear for d in dates])

    # Simulate 3-regime price series
    # Regime 0: low vol  (σ=0.02, μ=0)
    # Regime 1: mid vol  (σ=0.05, μ=0.001)
    # Regime 2: high vol (σ=0.10, μ=-0.001)
    true_regimes = np.zeros(n, dtype=int)
    # Inject regime switches
    switch_pts = sorted(rng.choice(n, size=20, replace=False))
    current_r  = 0
    for i, t in enumerate(switch_pts):
        current_r = (current_r + 1) % 3
        end = switch_pts[i+1] if i+1 < len(switch_pts) else n
        true_regimes[t:end] = current_r

    sigma_map = {0: 0.02, 1: 0.05, 2: 0.10}
    mu_map    = {0: 0.0,  1: 0.001, 2: -0.001}

    log_returns = np.array([
        rng.normal(mu_map[r], sigma_map[r]) for r in true_regimes
    ])
    prices = 80 * np.exp(np.cumsum(log_returns))
    prices_s = pd.Series(prices, index=dates, name="power_price")

    # Seasonal forward
    fwd_s = pd.Series(prices * (1 + 0.05*np.cos(2*np.pi*(doy-15)/365)),
                      index=dates)

    # Build features
    fe  = RegimeFeatureEngine(prices_s, forward_prices=fwd_s)
    obs = fe.build_observation_matrix()
    print(f"Observation matrix: {obs.shape[0]} rows × {obs.shape[1]} features")

    # Fit HMM
    cfg = HMMConfig(
        n_regimes    = 3,
        regime_names = ["Low-Vol/Contango", "Mid-Vol/Transition", "High-Vol/Backwardation"],
        regime_colours=["#1565c0", "#f9a825", "#c62828"],
    )
    detector = HMMRegimeDetector(cfg)
    detector.fit(obs)
    regimes  = detector.predict_regimes(obs)
    proba    = detector.predict_proba(obs)

    # Regime characterisation
    stats = detector.characterise_regimes(regimes, prices_s)
    print("\n=== Regime Characterisation ===")
    print(stats.to_string())

    print("\n=== Transition Matrix ===")
    print(detector.transition_matrix(regimes).to_string())

    print("\n=== Strategy Allocation by Regime ===")
    for k in range(3):
        alloc = detector.strategy_allocation(k)
        name  = cfg.regime_names[k] if k < len(cfg.regime_names) else f"R{k}"
        print(f"\n  {name}:")
        for strat, w in alloc.items():
            print(f"    {strat:25s}: {w:.0%}")

    fig = detector.plot(regimes, prices_s, proba)
    fig.savefig("regime_detection.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → regime_detection.png")
