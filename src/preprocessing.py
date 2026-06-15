from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class WindowedDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_dates: pd.DatetimeIndex
    prev_close_test: np.ndarray
    actual_close_test: np.ndarray
    scaler: MinMaxScaler
    target_index: int


def build_return_features(frame: pd.DataFrame, features) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in features:
        series = frame[col].astype(float)
        if col == "Volume":
            out[col] = np.log1p(series).diff()
        else:
            out[col] = np.log(series).diff()
    return out.dropna()


def _make_windows(matrix, target_col, window):
    x, y, idx = [], [], []
    for end in range(window, len(matrix)):
        x.append(matrix[end - window:end])
        y.append(matrix[end, target_col])
        idx.append(end)
    return np.asarray(x), np.asarray(y), idx


def prepare_datasets(frame, features, target, window, train_fraction, scaler=None):
    returns = build_return_features(frame, features)
    if len(returns) <= window + 2:
        raise ValueError(
            f"Not enough rows ({len(returns)}) for window {window}; need more history."
        )

    target_index = list(features).index(target)
    split = int(np.floor(len(returns) * train_fraction))
    if split <= window or split >= len(returns) - 1:
        raise ValueError("train_fraction leaves too few rows on one side of the split.")

    values = returns.to_numpy()
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(values[:split])
    elif scaler.n_features_in_ != values.shape[1]:
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features but data has {values.shape[1]}."
        )
    scaled = scaler.transform(values)

    x, y, ends = _make_windows(scaled, target_index, window)
    end_idx = np.asarray(ends)
    is_test = end_idx >= split

    close = frame["Close"].astype(float)
    close_on_return_index = close.loc[returns.index].to_numpy()
    prev_close = close_on_return_index[end_idx - 1]
    actual_close = close_on_return_index[end_idx]
    dates = returns.index[end_idx]

    return WindowedDataset(
        x_train=x[~is_test],
        y_train=y[~is_test],
        x_test=x[is_test],
        y_test=y[is_test],
        test_dates=dates[is_test],
        prev_close_test=prev_close[is_test],
        actual_close_test=actual_close[is_test],
        scaler=scaler,
        target_index=target_index,
    )


def inverse_return(scaled_values, scaler, target_index):
    scaled_values = np.asarray(scaled_values).reshape(-1)
    return (scaled_values - scaler.min_[target_index]) / scaler.scale_[target_index]


def reconstruct_price(prev_close, predicted_return):
    return np.asarray(prev_close) * np.exp(np.asarray(predicted_return))
