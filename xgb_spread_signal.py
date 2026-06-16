"""
XGBoost Spread Signal Classifier — Energy Markets
===================================================
Gradient-boosted tree model for classifying energy spread
trading signals: long, short, or flat.

Why XGBoost for energy trading signals?
    - Handles non-linear feature interactions natively
    - Robust to outliers and extreme values (common in energy)
    - Feature importance: identifies which signals matter most
    - Fast training and prediction: suitable for daily retraining
    - Works well with tabular data (unlike deep learning)
    - Interpretable via SHAP values

Problem formulation:
    INPUT:   Feature vector at time t (fundamentals, technicals, market data)
    OUTPUT:  Signal class {-1, 0, +1} or probability of each class

    Signal classes:
        +1 (Long):   Spread expected to widen → enter long
        -1 (Short):  Spread expected to narrow → enter short
         0 (Flat):   No clear edge → stay flat

    Label generation:
        Forward-looking label based on next N-day spread change:
            If spread_change > threshold → +1
            If spread_change < -threshold → -1
            Else → 0

Feature families:
    TECHNICAL:
        - Rolling z-score of spread (current deviation from mean)
        - Momentum: spread change over N days
        - RSI of spread
        - Bollinger band position

    FUNDAMENTAL:
        - Storage fill level vs seasonal normal
        - Net load deviation from forecast
        - Renewable penetration
        - Temperature deviation from seasonal normal
        - Gas price z-score

    MARKET STRUCTURE:
        - Forward curve slope (contango/backwardation)
        - Open interest (if available)
        - Volume ratio (today vs 30-day average)
        - Bid-ask spread (liquidity proxy)

    CROSS-ASSET:
        - EUA price z-score
        - FX rate (GBP/EUR for NBP-TTF)
        - LNG send-out level

Model pipeline:
    1. Feature engineering
    2. Label generation (forward-looking)
    3. Walk-forward cross-validation
    4. XGBoost training
    5. Signal extraction and backtest
    6. Feature importance analysis

Anti-lookahead:
    Labels use FUTURE data, so strict train/test separation is critical.
    Walk-forward validation: only train on past, predict on unseen future.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    # Fallback: use sklearn GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingClassifier


# ─── Feature engineering ──────────────────────────────────────────────────────

class SpreadFeatureEngine:
    """
    Feature engineering for spread signal classification.

    Parameters
    ----------
    spread          : pd.Series   Spread time series (e.g. spark spread, TTF-NCG)
    gas_prices      : pd.Series   Gas prices [€/MWh]
    eua_prices      : pd.Series   EUA prices [€/t]
    storage_fill    : pd.Series   Storage fill level [%]
    net_load        : pd.Series   Net load [MW]
    temperature     : pd.Series   Temperature [°C]
    """

    def __init__(
        self,
        spread: pd.Series,
        gas_prices: Optional[pd.Series]   = None,
        eua_prices: Optional[pd.Series]   = None,
        storage_fill: Optional[pd.Series] = None,
        net_load: Optional[pd.Series]     = None,
        temperature: Optional[pd.Series]  = None,
    ):
        self.spread  = spread.rename("spread")
        self.gas     = gas_prices
        self.eua     = eua_prices
        self.storage = storage_fill
        self.net_load= net_load
        self.temp    = temperature

    # ------------------------------------------------------------------
    # Technical features
    # ------------------------------------------------------------------

    def zscore_features(self, windows: List[int] = [10, 20, 60]) -> pd.DataFrame:
        """Rolling z-score of the spread for multiple windows."""
        df = pd.DataFrame(index=self.spread.index)
        for w in windows:
            m = self.spread.rolling(w).mean()
            s = self.spread.rolling(w).std().replace(0, np.nan)
            df[f"zscore_{w}"] = (self.spread - m) / s
        return df

    def momentum_features(self, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Spread momentum over multiple periods."""
        df = pd.DataFrame(index=self.spread.index)
        for p in periods:
            df[f"mom_{p}"]     = self.spread.diff(p)
            df[f"mom_pct_{p}"] = self.spread.pct_change(p) * 100
        return df

    def rsi(self, window: int = 14) -> pd.Series:
        """Relative Strength Index of the spread."""
        delta  = self.spread.diff()
        gain   = delta.clip(lower=0).rolling(window).mean()
        loss   = (-delta.clip(upper=0)).rolling(window).mean()
        rs     = gain / loss.replace(0, np.nan)
        rsi    = 100 - (100 / (1 + rs))
        rsi.name = f"rsi_{window}"
        return rsi

    def bollinger_features(self, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
        """Bollinger band position features."""
        mid  = self.spread.rolling(window).mean()
        std  = self.spread.rolling(window).std()
        upper= mid + n_std * std
        lower= mid - n_std * std
        bw   = (upper - lower) / mid.replace(0, np.nan)   # bandwidth
        pos  = (self.spread - lower) / (upper - lower).replace(0, np.nan)  # [0,1]
        return pd.DataFrame({
            "bb_position":  pos,
            "bb_bandwidth": bw,
            "bb_upper_dist": (upper - self.spread) / std.replace(0, np.nan),
            "bb_lower_dist": (self.spread - lower) / std.replace(0, np.nan),
        }, index=self.spread.index)

    def mean_reversion_speed(self, window: int = 30) -> pd.Series:
        """
        Rolling estimate of mean-reversion speed (AR(1) coefficient).
        Value close to 0 = fast reversion, close to 1 = random walk.
        """
        vals = pd.Series(np.nan, index=self.spread.index, name="ar1_coef")
        for i in range(window, len(self.spread)):
            window_data = self.spread.iloc[i-window:i].dropna()
            if len(window_data) < 10:
                continue
            y  = window_data.values[1:]
            x  = window_data.values[:-1]
            if x.std() == 0:
                continue
            b = np.cov(x, y)[0, 1] / np.var(x)
            vals.iloc[i] = b
        return vals

    # ------------------------------------------------------------------
    # Fundamental features
    # ------------------------------------------------------------------

    def fundamental_features(self) -> pd.DataFrame:
        """Fundamental signal features from market data."""
        idx = self.spread.index
        df  = pd.DataFrame(index=idx)

        if self.gas is not None:
            gas_m = self.gas.rolling(20).mean()
            gas_s = self.gas.rolling(20).std().replace(0, np.nan)
            df["gas_zscore"]  = (self.gas - gas_m) / gas_s
            df["gas_mom_5"]   = self.gas.diff(5)
            df["gas_log_ret"] = np.log(self.gas / self.gas.shift(1))

        if self.eua is not None:
            eua_m = self.eua.rolling(20).mean()
            eua_s = self.eua.rolling(20).std().replace(0, np.nan)
            df["eua_zscore"]  = (self.eua - eua_m) / eua_s
            df["eua_mom_5"]   = self.eua.diff(5)

        if self.storage is not None:
            doy     = idx.dayofyear
            seas_m  = self.storage.groupby(doy).transform("mean")
            seas_s  = self.storage.groupby(doy).transform("std").replace(0, np.nan)
            df["storage_zscore"]   = (self.storage - seas_m) / seas_s
            df["storage_level"]    = self.storage
            df["storage_mom_5"]    = self.storage.diff(5)

        if self.net_load is not None:
            nl_m = self.net_load.rolling(20).mean()
            nl_s = self.net_load.rolling(20).std().replace(0, np.nan)
            df["netload_zscore"] = (self.net_load - nl_m) / nl_s

        if self.temp is not None:
            doy   = idx.dayofyear
            t_m   = self.temp.groupby(doy).transform("mean")
            df["temp_deviation"] = self.temp - t_m

        return df

    # ------------------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------------------

    def calendar_features(self) -> pd.DataFrame:
        """Time-based features."""
        idx = self.spread.index
        df  = pd.DataFrame(index=idx)
        df["month"]        = idx.month
        df["quarter"]      = idx.quarter
        df["dayofweek"]    = idx.dayofweek
        df["is_monday"]    = (idx.dayofweek == 0).astype(int)
        df["is_friday"]    = (idx.dayofweek == 4).astype(int)
        df["is_winter"]    = idx.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
        # Cyclical encoding
        df["sin_month"]    = np.sin(2*np.pi*idx.month/12)
        df["cos_month"]    = np.cos(2*np.pi*idx.month/12)
        df["sin_dow"]      = np.sin(2*np.pi*idx.dayofweek/7)
        df["cos_dow"]      = np.cos(2*np.pi*idx.dayofweek/7)
        return df

    # ------------------------------------------------------------------
    # Full feature matrix
    # ------------------------------------------------------------------

    def build_features(self) -> pd.DataFrame:
        """Combine all feature groups into a single DataFrame."""
        dfs = [
            self.zscore_features(),
            self.momentum_features(),
            self.rsi(14).to_frame(),
            self.rsi(7).to_frame(),
            self.bollinger_features(),
            self.fundamental_features(),
            self.calendar_features(),
        ]
        feat = pd.concat(dfs, axis=1)
        feat = feat.replace([np.inf, -np.inf], np.nan)
        return feat


# ─── Label generation ────────────────────────────────────────────────────────

def generate_labels(
    spread: pd.Series,
    horizon: int   = 5,
    threshold: float = 0.5,
) -> pd.Series:
    """
    Generate forward-looking labels for spread trading signals.

    Labels:
        +1 if spread rises by > threshold over next `horizon` days
        -1 if spread falls by > threshold over next `horizon` days
         0 otherwise

    Parameters
    ----------
    spread    : pd.Series   Spread time series
    horizon   : int         Forward window [days]
    threshold : float       Minimum move to generate signal [€/MWh or spread units]
    """
    fwd_change = spread.shift(-horizon) - spread
    labels     = pd.Series(0, index=spread.index, name="signal")
    labels[fwd_change >  threshold] =  1
    labels[fwd_change < -threshold] = -1
    return labels


# ─── XGBoost classifier ──────────────────────────────────────────────────────

@dataclass
class XGBConfig:
    """XGBoost model configuration."""
    n_estimators: int   = 300
    max_depth: int      = 5
    learning_rate: float= 0.05
    subsample: float    = 0.80
    colsample_bytree: float = 0.70
    min_child_weight: int   = 5
    gamma: float        = 0.1
    reg_alpha: float    = 0.1
    reg_lambda: float   = 1.0
    scale_pos_weight: float = 1.0
    n_cv_splits: int    = 5
    horizon: int        = 5
    label_threshold: float = 0.5
    flat_zone: float    = 0.35     # probability below which signal = flat


class XGBSpreadClassifier:
    """
    XGBoost-based spread signal classifier.

    Parameters
    ----------
    config : XGBConfig
    """

    def __init__(self, config: Optional[XGBConfig] = None):
        self.cfg     = config or XGBConfig()
        self.model   = None
        self.scaler  = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def _build_model(self):
        """Build XGBoost or fallback classifier."""
        cfg = self.cfg
        if XGB_AVAILABLE:
            return xgb.XGBClassifier(
                n_estimators    = cfg.n_estimators,
                max_depth       = cfg.max_depth,
                learning_rate   = cfg.learning_rate,
                subsample       = cfg.subsample,
                colsample_bytree= cfg.colsample_bytree,
                min_child_weight= cfg.min_child_weight,
                gamma           = cfg.gamma,
                reg_alpha       = cfg.reg_alpha,
                reg_lambda      = cfg.reg_lambda,
                use_label_encoder=False,
                eval_metric     = "mlogloss",
                random_state    = 42,
                n_jobs          = -1,
            )
        else:
            return GradientBoostingClassifier(
                n_estimators = cfg.n_estimators,
                max_depth    = cfg.max_depth,
                learning_rate= cfg.learning_rate,
                subsample    = cfg.subsample,
                random_state = 42,
            )

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> "XGBSpreadClassifier":
        """Fit the model on training data."""
        self.feature_names = list(features.columns)
        common = features.index.intersection(labels.index)
        X = features.loc[common].fillna(0)
        y = labels.loc[common]

        # Shift labels to {0, 1, 2} for XGBoost
        y_mapped = y.map({-1: 0, 0: 1, 1: 2})

        X_sc = self.scaler.fit_transform(X)
        self.model = self._build_model()
        self.model.fit(X_sc, y_mapped)
        self.is_fitted = True
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        X    = features.reindex(columns=self.feature_names).fillna(0)
        X_sc = self.scaler.transform(X)
        prob = self.model.predict_proba(X_sc)
        return pd.DataFrame(prob, index=features.index,
                            columns=["p_short", "p_flat", "p_long"])

    def predict_signal(self, features: pd.DataFrame) -> pd.Series:
        """
        Return trading signals {-1, 0, +1}.
        Applies flat zone: signal = 0 if max probability < flat_zone.
        """
        proba  = self.predict_proba(features)
        preds  = proba.idxmax(axis=1).map({"p_short": -1, "p_flat": 0, "p_long": 1})
        max_p  = proba.max(axis=1)
        preds[max_p < self.cfg.flat_zone] = 0
        return preds.rename("signal")

    def feature_importance(self) -> pd.Series:
        """Return normalised feature importance."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        if XGB_AVAILABLE:
            imp = self.model.feature_importances_
        else:
            imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names,
                         name="importance").sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Walk-forward backtest
    # ------------------------------------------------------------------

    def walk_forward_backtest(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        spread: pd.Series,
        min_train: int = 200,
        step: int      = 20,
    ) -> pd.DataFrame:
        """
        Walk-forward backtest with expanding training window.
        """
        records = []
        n       = len(features)

        for end_train in range(min_train, n - self.cfg.horizon, step):
            # Fit on training data
            tr_feat = features.iloc[:end_train]
            tr_lbl  = labels.iloc[:end_train]
            try:
                self.fit(tr_feat, tr_lbl)
            except Exception:
                continue

            # Predict on test window
            test_end  = min(end_train + step, n - self.cfg.horizon)
            te_feat   = features.iloc[end_train:test_end]
            if len(te_feat) == 0:
                continue

            te_sig    = self.predict_signal(te_feat)
            te_proba  = self.predict_proba(te_feat)

            for i, (dt, sig) in enumerate(te_sig.items()):
                actual_lbl = labels.get(dt, 0)
                sp_chg     = spread.shift(-self.cfg.horizon).get(dt, np.nan) - spread.get(dt, np.nan)
                pnl        = float(sig) * float(sp_chg) if not np.isnan(sp_chg) else 0.0
                records.append({
                    "date":      dt,
                    "signal":    sig,
                    "actual_lbl":actual_lbl,
                    "correct":   sig == actual_lbl,
                    "spread_chg":sp_chg,
                    "pnl":       pnl,
                    "p_long":    te_proba.loc[dt, "p_long"],
                    "p_short":   te_proba.loc[dt, "p_short"],
                    "p_flat":    te_proba.loc[dt, "p_flat"],
                })

        return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot_backtest(
        self, backtest: pd.DataFrame, spread: pd.Series, figsize=(14, 13)
    ) -> plt.Figure:
        if backtest.empty:
            raise ValueError("Empty backtest.")
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle("XGBoost Spread Signal Classifier — Backtest",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(spread.index, spread.values, color="#333", lw=0.8, label="Spread")
        longs  = backtest[backtest["signal"] == 1]
        shorts = backtest[backtest["signal"] == -1]
        ax.scatter(longs.index,  spread.reindex(longs.index),  s=20, color="green", zorder=5, label="Long signal")
        ax.scatter(shorts.index, spread.reindex(shorts.index), s=20, color="red",   zorder=5, label="Short signal")
        ax.legend(fontsize=8); ax.set_ylabel("Spread"); ax.set_title("Spread & Signals", fontsize=10)

        ax = axes[1]
        ax.fill_between(backtest.index, backtest["p_long"],  0, color="green", alpha=0.5, label="P(Long)")
        ax.fill_between(backtest.index, -backtest["p_short"], 0, color="red",  alpha=0.5, label="P(Short)")
        ax.axhline(self.cfg.flat_zone, color="orange", lw=0.7, ls="--", label="Flat zone")
        ax.legend(fontsize=8); ax.set_ylabel("Probability")
        ax.set_title("Signal Probabilities", fontsize=10)

        ax = axes[2]
        correct = backtest["correct"].astype(int)
        ax.fill_between(backtest.index, correct.rolling(20).mean() * 100, 50,
                        where=correct.rolling(20).mean() > 0.5, color="green", alpha=0.4)
        ax.fill_between(backtest.index, correct.rolling(20).mean() * 100, 50,
                        where=correct.rolling(20).mean() <= 0.5, color="red", alpha=0.4)
        ax.axhline(50, color="black", lw=0.5, ls="--")
        ax.set_ylabel("%"); ax.set_title("Rolling Accuracy (20-day)", fontsize=10)

        ax = axes[3]
        cum_pnl = backtest["pnl"].cumsum()
        ax.fill_between(backtest.index, cum_pnl, 0,
                        where=cum_pnl >= 0, color="green", alpha=0.4)
        ax.fill_between(backtest.index, cum_pnl, 0,
                        where=cum_pnl < 0,  color="red",   alpha=0.4)
        ax.plot(backtest.index, cum_pnl, color="black", lw=0.8)
        ax.set_ylabel("€/MWh cumul."); ax.set_title("Cumulative P&L", fontsize=10)
        ax.set_xlabel("Date")

        plt.tight_layout()
        return fig

    def plot_feature_importance(
        self, top_n: int = 20, figsize=(10, 7)
    ) -> plt.Figure:
        """Bar chart of top feature importances."""
        imp = self.feature_importance().head(top_n)
        fig, ax = plt.subplots(figsize=figsize)
        imp[::-1].plot.barh(ax=ax, color="#1565c0", alpha=0.8)
        ax.set_title(f"XGBoost Feature Importance — Top {top_n}", fontsize=11)
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        return fig


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng   = np.random.default_rng(42)
    n     = 800
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    doy   = np.array([d.dayofyear for d in dates])

    # Synthetic spark spread
    gas     = np.clip(40 + np.cumsum(rng.normal(0, 1.2, n)), 15, 200)
    power   = np.clip(gas/0.5 + rng.normal(0,5,n) + 5, 20, 400)
    eua     = np.clip(65 + np.cumsum(rng.normal(0, 0.8, n)), 20, 130)
    storage = np.clip(55 + 30*np.sin(2*np.pi*(doy-90)/365) + rng.normal(0,4,n), 5, 100)
    nl      = np.clip(45000 + 12000*np.cos(2*np.pi*(doy-15)/365) + rng.normal(0,2000,n), 25000, 70000)
    temp    = 10 - 12*np.cos(2*np.pi*(doy-15)/365) + rng.normal(0,3,n)

    spark = power - gas/0.5 - eua*0.74 - 2.0
    spark_s = pd.Series(spark, index=dates, name="spark_spread")

    fe = SpreadFeatureEngine(
        spread       = spark_s,
        gas_prices   = pd.Series(gas, index=dates),
        eua_prices   = pd.Series(eua, index=dates),
        storage_fill = pd.Series(storage, index=dates),
        net_load     = pd.Series(nl, index=dates),
        temperature  = pd.Series(temp, index=dates),
    )
    features = fe.build_features()
    labels   = generate_labels(spark_s, horizon=5, threshold=1.0)

    print(f"Feature matrix: {features.shape[0]} rows × {features.shape[1]} features")
    print(f"Label distribution:\n{labels.value_counts().to_string()}")

    cfg   = XGBConfig(n_estimators=200, max_depth=4, horizon=5, flat_zone=0.38)
    model = XGBSpreadClassifier(cfg)

    print("\nRunning walk-forward backtest...")
    bt = model.walk_forward_backtest(features, labels, spark_s, min_train=200, step=15)

    if not bt.empty:
        accuracy = bt["correct"].mean()
        total_pnl= bt["pnl"].sum()
        sharpe   = bt["pnl"].mean() / bt["pnl"].std() * np.sqrt(252) if bt["pnl"].std() > 0 else 0
        print(f"\n=== XGBoost Classifier — Backtest Results ===")
        print(f"  Overall accuracy:  {accuracy:.1%}")
        print(f"  Total P&L:         {total_pnl:.2f} €/MWh")
        print(f"  Sharpe ratio:      {sharpe:.3f}")
        print(f"  Signals long:      {(bt['signal']==1).sum()}")
        print(f"  Signals short:     {(bt['signal']==-1).sum()}")
        print(f"  Signals flat:      {(bt['signal']==0).sum()}")

        fig1 = model.plot_backtest(bt, spark_s)
        fig1.savefig("xgb_spread_signal.png", dpi=150, bbox_inches="tight")

        imp_fig = model.plot_feature_importance(top_n=15)
        imp_fig.savefig("xgb_feature_importance.png", dpi=150, bbox_inches="tight")
        print("\nCharts saved.")
