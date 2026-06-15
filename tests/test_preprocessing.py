from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_return_features,
    inverse_return,
    prepare_datasets,
    reconstruct_price,
)

FEATURES = ("Open", "High", "Low", "Close", "Volume")


def _frame(n=260, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    steps = rng.normal(0, 0.01, n).cumsum()
    close = 100 * np.exp(steps)
    frame = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1e5, 5e5, n).astype(float),
        },
        index=dates,
    )
    frame.index.name = "Date"
    return frame


def test_return_features_drop_first_row():
    frame = _frame(50)
    returns = build_return_features(frame, FEATURES)
    assert len(returns) == len(frame) - 1
    assert list(returns.columns) == list(FEATURES)


def test_window_shapes_align():
    frame = _frame()
    data = prepare_datasets(frame, FEATURES, "Close", 30, 0.65)
    assert data.x_train.shape[1:] == (30, len(FEATURES))
    assert len(data.x_test) == len(data.y_test) == len(data.test_dates)
    assert len(data.prev_close_test) == len(data.actual_close_test) == len(data.x_test)


def test_chronological_split_is_strict():
    frame = _frame()
    data = prepare_datasets(frame, FEATURES, "Close", 30, 0.65)
    assert data.test_dates.is_monotonic_increasing
    assert len(data.x_train) > 0 and len(data.x_test) > 0


def test_scaler_fit_on_train_only():
    frame = _frame()
    data = prepare_datasets(frame, FEATURES, "Close", 30, 0.65)
    returns = build_return_features(frame, FEATURES).to_numpy()
    split = int(np.floor(len(returns) * 0.65))
    expected_max = returns[:split].max(axis=0)
    assert np.allclose(data.scaler.data_max_, expected_max)


def test_provided_scaler_is_reused():
    frame = _frame()
    first = prepare_datasets(frame, FEATURES, "Close", 30, 0.65)
    second = prepare_datasets(frame, FEATURES, "Close", 30, 0.65, scaler=first.scaler)
    assert second.scaler is first.scaler


def test_mismatched_scaler_rejected():
    frame = _frame()
    full = prepare_datasets(frame, FEATURES, "Close", 30, 0.65)
    with pytest.raises(ValueError):
        prepare_datasets(frame, ("Close",), "Close", 30, 0.65, scaler=full.scaler)


def test_inverse_return_round_trip():
    frame = _frame()
    data = prepare_datasets(frame, FEATURES, "Close", 30, 0.65)
    recovered = inverse_return(data.y_test, data.scaler, data.target_index)
    returns = build_return_features(frame, FEATURES)["Close"].to_numpy()
    assert recovered.max() <= returns.max() + 1e-6


def test_reconstruct_price_matches_actual_with_true_return():
    frame = _frame()
    data = prepare_datasets(frame, FEATURES, "Close", 30, 0.65)
    true_return = inverse_return(data.y_test, data.scaler, data.target_index)
    rebuilt = reconstruct_price(data.prev_close_test, true_return)
    assert np.allclose(rebuilt, data.actual_close_test, rtol=1e-6)


def test_too_few_rows_raises():
    frame = _frame(35)
    with pytest.raises(ValueError):
        prepare_datasets(frame, FEATURES, "Close", 30, 0.65)
