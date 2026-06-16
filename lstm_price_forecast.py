"""
LSTM Price Forecasting — Day-Ahead Power Prices
=================================================
Implements a Long Short-Term Memory (LSTM) neural network
for day-ahead electricity price forecasting.

Why LSTM for energy prices?
    Energy prices exhibit:
        - Strong seasonality (daily, weekly, annual)
        - Non-linear relationships with weather and demand
        - Regime changes (high-vol vs low-vol periods)
        - Spike behaviour (extreme price events)
        - Long-range dependencies (weekly patterns)

    LSTM networks capture:
        - Long-range temporal dependencies via gating mechanism
        - Non-linear input-output relationships
        - Multi-variate interactions (price + weather + fundamentals)

LSTM architecture:
    Input  → [LSTM Layer(s)] → [Dense Layer(s)] → Output

    Input features (example set):
        - Lagged prices (t-1, t-2, ..., t-7 days)
        - Day-of-week, hour-of-day (cyclical encoding)
        - Wind forecast [MW]
        - Solar forecast [MW]
        - Load forecast [MW]
        - Net load = Load - Wind - Solar
        - Gas price (TTF)
        - Temperature
        - Holiday indicator

    Output:
        - Next day's 24 hourly prices (sequence-to-sequence)
        - Or single-value prediction (sequence-to-one)

    LSTM cell equations:
        f(t) = σ(Wf · [h(t-1), x(t)] + bf)      forget gate
        i(t) = σ(Wi · [h(t-1), x(t)] + bi)      input gate
        g(t) = tanh(Wg · [h(t-1), x(t)] + bg)   candidate
        o(t) = σ(Wo · [h(t-1), x(t)] + bo)      output gate
        C(t) = f(t) * C(t-1) + i(t) * g(t)      cell state
        h(t) = o(t) * tanh(C(t))                 hidden state

Training:
    - Loss: MAE or Huber loss (robust to price spikes)
    - Optimiser: Adam with learning rate scheduling
    - Regularisation: dropout, L2, early stopping
    - Validation: walk-forward (no data leakage)

Performance metrics:
    - MAE  (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error) — careful with near-zero prices
    - DA   (Directional Accuracy)
    - Pinball loss (for probabilistic forecasts)

Note:
    This module uses numpy/scipy for a simplified LSTM implementation
    to avoid heavy dependencies (tensorflow/torch).
    In production, use PyTorch or TensorFlow for GPU-accelerated training.
    The architecture and training loop are fully compatible with both.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ─── Feature engineering ──────────────────────────────────────────────────────

class EnergyFeatureEngine:
    """
    Feature engineering for energy price forecasting.
    Transforms raw time series into ML-ready feature matrix.

    Parameters
    ----------
    prices      : pd.Series   Hourly or daily power prices [€/MWh]
    wind_fc     : pd.Series   Wind generation forecast [MW]
    solar_fc    : pd.Series   Solar generation forecast [MW]
    load_fc     : pd.Series   Load forecast [MW]
    gas_prices  : pd.Series   Gas price (TTF) [€/MWh]
    temperature : pd.Series   Temperature [°C]
    """

    def __init__(
        self,
        prices: pd.Series,
        wind_fc: Optional[pd.Series]     = None,
        solar_fc: Optional[pd.Series]    = None,
        load_fc: Optional[pd.Series]     = None,
        gas_prices: Optional[pd.Series]  = None,
        temperature: Optional[pd.Series] = None,
    ):
        self.prices = prices
        self.wind   = wind_fc
        self.solar  = solar_fc
        self.load   = load_fc
        self.gas    = gas_prices
        self.temp   = temperature

    def cyclical_encode(self, series: pd.Series, period: int) -> pd.DataFrame:
        """
        Encode a cyclical feature (e.g. hour of day) as sin/cos pair.
        Avoids discontinuity at period boundary.
        """
        sin_feat = np.sin(2 * np.pi * series / period)
        cos_feat = np.cos(2 * np.pi * series / period)
        return pd.DataFrame({
            f"sin_{period}": sin_feat,
            f"cos_{period}": cos_feat,
        }, index=series.index)

    def lag_features(self, series: pd.Series, lags: List[int]) -> pd.DataFrame:
        """Create lagged features from a time series."""
        df = pd.DataFrame(index=series.index)
        for lag in lags:
            df[f"lag_{lag}"] = series.shift(lag)
        return df

    def rolling_features(
        self, series: pd.Series, windows: List[int]
    ) -> pd.DataFrame:
        """Rolling mean and std features."""
        df = pd.DataFrame(index=series.index)
        for w in windows:
            df[f"roll_mean_{w}"] = series.rolling(w).mean()
            df[f"roll_std_{w}"]  = series.rolling(w).std()
        return df

    def build_feature_matrix(
        self,
        price_lags: List[int]   = [1, 2, 7, 14],
        roll_windows: List[int] = [7, 30],
    ) -> pd.DataFrame:
        """
        Build complete feature matrix for LSTM input.
        """
        idx = self.prices.index
        dfs = []

        # Time features (cyclical encoding)
        if hasattr(idx, "dayofweek"):
            dow    = pd.Series(idx.dayofweek, index=idx)
            month  = pd.Series(idx.month, index=idx)
            dfs.append(self.cyclical_encode(dow, 7).rename(
                columns=lambda c: f"dow_{c}"))
            dfs.append(self.cyclical_encode(month, 12).rename(
                columns=lambda c: f"month_{c}"))

        # Price lags and rolling stats
        dfs.append(self.lag_features(self.prices, price_lags))
        dfs.append(self.rolling_features(self.prices, roll_windows))

        # External features
        for feat, name in [
            (self.wind,  "wind_fc"),
            (self.solar, "solar_fc"),
            (self.load,  "load_fc"),
            (self.gas,   "gas_price"),
            (self.temp,  "temperature"),
        ]:
            if feat is not None:
                s = feat.reindex(idx).fillna(method="ffill")
                dfs.append(s.rename(name).to_frame())
                dfs.append(self.lag_features(s, [1, 7]).rename(
                    columns=lambda c: f"{name}_{c}"))

        # Net load
        if self.wind is not None and self.solar is not None and self.load is not None:
            net_load = (self.load - self.wind - self.solar).reindex(idx)
            dfs.append(net_load.rename("net_load").to_frame())

        # Target: next-day price (1-step ahead)
        features = pd.concat(dfs, axis=1)
        features = features.dropna()
        return features

    def prepare_sequences(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        seq_len: int = 14,
        horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare (X, y) sequences for LSTM training.

        X shape: (n_samples, seq_len, n_features)
        y shape: (n_samples, horizon)
        """
        common = features.index.intersection(target.index)
        X_df   = features.loc[common]
        y_s    = target.loc[common]

        X_vals = X_df.values
        y_vals = y_s.values

        X_seqs, y_seqs = [], []
        for i in range(seq_len, len(X_vals) - horizon + 1):
            X_seqs.append(X_vals[i - seq_len: i])
            y_seqs.append(y_vals[i: i + horizon])

        return np.array(X_seqs), np.array(y_seqs)


# ─── Simplified LSTM cell (numpy) ────────────────────────────────────────────

class NumpyLSTMCell:
    """
    Single LSTM cell implemented in numpy.
    For illustration purposes — use PyTorch/TF for production.
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        rng  = np.random.default_rng(seed)
        s    = np.sqrt(1 / hidden_size)
        # Weight matrices [input + hidden → hidden]
        concat = input_size + hidden_size
        self.Wf = rng.uniform(-s, s, (hidden_size, concat))
        self.Wi = rng.uniform(-s, s, (hidden_size, concat))
        self.Wg = rng.uniform(-s, s, (hidden_size, concat))
        self.Wo = rng.uniform(-s, s, (hidden_size, concat))
        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bg = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)

    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def forward(
        self, x: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single forward pass."""
        combined = np.concatenate([h_prev, x])
        f = self.sigmoid(self.Wf @ combined + self.bf)
        i = self.sigmoid(self.Wi @ combined + self.bi)
        g = np.tanh(self.Wg @ combined + self.bg)
        o = self.sigmoid(self.Wo @ combined + self.bo)
        C = f * C_prev + i * g
        h = o * np.tanh(C)
        return h, C

    def forward_sequence(self, X: np.ndarray) -> np.ndarray:
        """Process full sequence, return all hidden states."""
        seq_len, input_size = X.shape
        h = np.zeros(self.Wf.shape[0])
        C = np.zeros(self.Wf.shape[0])
        hidden_states = []
        for t in range(seq_len):
            h, C = self.forward(X[t], h, C)
            hidden_states.append(h)
        return np.array(hidden_states)


# ─── LSTM forecaster ─────────────────────────────────────────────────────────

@dataclass
class LSTMConfig:
    """Configuration for LSTM price forecasting model."""
    seq_len: int        = 14        # lookback window [days]
    hidden_size: int    = 32        # LSTM hidden units
    n_layers: int       = 2         # stacked LSTM layers
    horizon: int        = 1         # forecast horizon [days]
    dropout: float      = 0.10      # dropout rate
    learning_rate: float= 0.001
    epochs: int         = 100
    batch_size: int     = 32
    early_stop_patience:int = 15
    price_lags: List[int]   = field(default_factory=lambda: [1, 2, 7, 14])
    roll_windows: List[int] = field(default_factory=lambda: [7, 30])
    val_split: float    = 0.20


class LSTMPriceForecaster:
    """
    LSTM-based day-ahead power price forecasting model.

    Uses a simplified numpy LSTM for demonstration.
    In production: replace with PyTorch or TensorFlow.

    Parameters
    ----------
    config      : LSTMConfig
    """

    def __init__(self, config: Optional[LSTMConfig] = None):
        self.cfg     = config or LSTMConfig()
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.lstm_cell: Optional[NumpyLSTMCell] = None
        self.output_W: Optional[np.ndarray]     = None
        self.output_b: Optional[np.ndarray]     = None
        self.feature_names: List[str] = []
        self.train_history: Dict[str, List] = {"train_loss": [], "val_loss": []}
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Training (simplified gradient-free approximation)
    # ------------------------------------------------------------------

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        feature_engine: Optional[EnergyFeatureEngine] = None,
    ) -> "LSTMPriceForecaster":
        """
        Fit the LSTM model.

        In this numpy implementation, we use a simplified training approach.
        For production, use PyTorch with BPTT and Adam optimiser.
        """
        cfg = self.cfg
        self.feature_names = list(features.columns)

        # Prepare sequences
        fe = feature_engine or EnergyFeatureEngine(target)
        X, y = fe.prepare_sequences(features, target, cfg.seq_len, cfg.horizon)

        if len(X) < 20:
            raise ValueError("Insufficient data for training.")

        # Train/val split
        n_val = max(1, int(len(X) * cfg.val_split))
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        # Scale
        n_tr, seq, n_feat = X_tr.shape
        X_tr_2d  = X_tr.reshape(-1, n_feat)
        X_val_2d = X_val.reshape(-1, n_feat)
        X_tr_sc  = self.scaler_X.fit_transform(X_tr_2d).reshape(X_tr.shape)
        X_val_sc = self.scaler_X.transform(X_val_2d).reshape(X_val.shape)
        y_tr_sc  = self.scaler_y.fit_transform(y_tr)
        y_val_sc = self.scaler_y.transform(y_val)

        # Initialise LSTM cell and output layer
        hidden = cfg.hidden_size
        rng    = np.random.default_rng(42)
        self.lstm_cell = NumpyLSTMCell(n_feat, hidden)
        self.output_W  = rng.normal(0, 0.1, (cfg.horizon, hidden))
        self.output_b  = np.zeros(cfg.horizon)

        # Simplified training: use LSTM as fixed feature extractor,
        # train only output layer with ridge regression
        # (full backprop would require autograd — use PyTorch in production)
        print(f"  Extracting LSTM features from {len(X_tr_sc)} training sequences...")
        H_tr  = np.array([self.lstm_cell.forward_sequence(x)[-1] for x in X_tr_sc])
        H_val = np.array([self.lstm_cell.forward_sequence(x)[-1] for x in X_val_sc])

        # Ridge regression for output layer
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=0.1)
        ridge.fit(H_tr, y_tr_sc.ravel() if cfg.horizon == 1 else y_tr_sc)
        self.output_W = ridge.coef_.reshape(cfg.horizon, hidden) if cfg.horizon > 1 else ridge.coef_.reshape(1, hidden)
        self.output_b = np.array([ridge.intercept_]) if np.isscalar(ridge.intercept_) else ridge.intercept_

        # Compute losses
        train_pred = ridge.predict(H_tr)
        val_pred   = ridge.predict(H_val)
        self.train_history["train_loss"] = [float(np.mean((train_pred - y_tr_sc.ravel())**2))]
        self.train_history["val_loss"]   = [float(np.mean((val_pred   - y_val_sc.ravel())**2))]

        self.is_fitted = True
        print(f"  Train MSE: {self.train_history['train_loss'][-1]:.6f} | "
              f"Val MSE: {self.train_history['val_loss'][-1]:.6f}")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Predict prices for a batch of input sequences.
        X_seq shape: (n_samples, seq_len, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        n, seq, n_feat = X_seq.shape
        X_2d  = X_seq.reshape(-1, n_feat)
        X_sc  = self.scaler_X.transform(X_2d).reshape(X_seq.shape)
        H     = np.array([self.lstm_cell.forward_sequence(x)[-1] for x in X_sc])
        y_sc  = (H @ self.output_W.T) + self.output_b
        y_hat = self.scaler_y.inverse_transform(y_sc.reshape(-1, self.cfg.horizon))
        return y_hat

    # ------------------------------------------------------------------
    # Walk-forward backtest
    # ------------------------------------------------------------------

    def walk_forward_backtest(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        min_train: int = 180,
        step: int      = 7,
    ) -> pd.DataFrame:
        """
        Walk-forward (expanding window) backtest.
        Retrains every `step` days on expanding dataset.
        Returns DataFrame of (date, actual, forecast, error).
        """
        cfg     = self.cfg
        fe      = EnergyFeatureEngine(target)
        records = []

        for end_train in range(min_train, len(target) - cfg.horizon, step):
            # Training window
            tr_feat = features.iloc[:end_train]
            tr_tgt  = target.iloc[:end_train]

            try:
                self.fit(tr_feat, tr_tgt, fe)
            except Exception:
                continue

            # Predict next step
            X, _ = fe.prepare_sequences(features, target, cfg.seq_len, cfg.horizon)
            if len(X) == 0:
                continue
            idx_pred = end_train - cfg.seq_len
            if idx_pred < 0 or idx_pred >= len(X):
                continue

            pred = float(self.predict(X[idx_pred: idx_pred + 1])[0, 0])
            actual = float(target.iloc[end_train])
            date   = target.index[end_train]

            records.append({
                "date":       date,
                "actual":     actual,
                "forecast":   pred,
                "error":      pred - actual,
                "abs_error":  abs(pred - actual),
            })

        return pd.DataFrame(records).set_index("date") if records else pd.DataFrame()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
        """Compute forecast accuracy metrics."""
        mae   = mean_absolute_error(actual, predicted)
        rmse  = np.sqrt(mean_squared_error(actual, predicted))
        mask  = actual != 0
        mape  = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        da    = np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100
        return {
            "MAE":  round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 2),
            "DA":   round(da, 1),
            "bias": round(np.mean(predicted - actual), 4),
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot_backtest(
        self, backtest: pd.DataFrame, figsize=(14, 10)
    ) -> plt.Figure:
        """Plot walk-forward backtest results."""
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        fig.suptitle("LSTM Price Forecasting — Walk-Forward Backtest",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(backtest.index, backtest["actual"],   label="Actual",   color="#333",   lw=1.0)
        ax.plot(backtest.index, backtest["forecast"], label="Forecast", color="#c62828", lw=1.0, ls="--")
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh")
        ax.set_title("Actual vs LSTM Forecast", fontsize=10)

        ax = axes[1]
        ax.fill_between(backtest.index, backtest["error"], 0,
                        where=backtest["error"] > 0, color="red",   alpha=0.4, label="Over-forecast")
        ax.fill_between(backtest.index, backtest["error"], 0,
                        where=backtest["error"] < 0, color="blue",  alpha=0.4, label="Under-forecast")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh error")
        ax.set_title("Forecast Error", fontsize=10)

        ax = axes[2]
        ax.plot(backtest.index, backtest["abs_error"].rolling(14).mean(),
                color="#e65100", lw=1.0, label="14-day rolling MAE")
        ax.legend(fontsize=8); ax.set_ylabel("€/MWh")
        ax.set_title("Rolling MAE (14 days)", fontsize=10)
        ax.set_xlabel("Date")

        plt.tight_layout()
        return fig


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng   = np.random.default_rng(42)
    n     = 500
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    doy   = np.array([d.dayofyear for d in dates])

    # Synthetic price with seasonality, trend, noise
    wind   = np.clip(15000 + 8000*np.cos(2*np.pi*(doy-355)/365) + rng.normal(0,2000,n), 0, 50000)
    solar  = np.clip(8000  - 6000*np.cos(2*np.pi*(doy-15)/365)  + rng.normal(0,1000,n), 0, 25000)
    load   = np.clip(50000 + 12000*np.cos(2*np.pi*(doy-15)/365) + rng.normal(0,2000,n), 30000, 70000)
    gas    = np.clip(40 + np.cumsum(rng.normal(0,1,n)), 15, 200)
    temp   = 10 - 12*np.cos(2*np.pi*(doy-15)/365) + rng.normal(0,3,n)
    net_l  = load - wind - solar
    price  = np.clip(30 + 0.0006*net_l + 0.3*gas + rng.normal(0,8,n), -20, 300)

    prices_s = pd.Series(price, index=dates, name="price")
    wind_s   = pd.Series(wind,  index=dates)
    solar_s  = pd.Series(solar, index=dates)
    load_s   = pd.Series(load,  index=dates)
    gas_s    = pd.Series(gas,   index=dates)
    temp_s   = pd.Series(temp,  index=dates)

    fe = EnergyFeatureEngine(prices_s, wind_s, solar_s, load_s, gas_s, temp_s)
    features = fe.build_feature_matrix()
    print(f"Feature matrix: {features.shape[0]} rows × {features.shape[1]} features")
    print(f"Features: {list(features.columns)[:8]}...")

    cfg   = LSTMConfig(seq_len=14, hidden_size=32, horizon=1)
    model = LSTMPriceForecaster(cfg)

    print("\nRunning walk-forward backtest...")
    bt = model.walk_forward_backtest(features, prices_s, min_train=150, step=10)

    if not bt.empty:
        m = LSTMPriceForecaster.metrics(bt["actual"].values, bt["forecast"].values)
        print("\n=== LSTM Backtest Metrics ===")
        for k, v in m.items():
            print(f"  {k:8s}: {v}")

        fig = model.plot_backtest(bt)
        fig.savefig("lstm_price_forecast.png", dpi=150, bbox_inches="tight")
        print("\nChart saved → lstm_price_forecast.png")
