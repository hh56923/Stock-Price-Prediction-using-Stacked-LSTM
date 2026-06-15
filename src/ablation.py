from __future__ import annotations

import argparse
import json
import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .config import load_config
from .data import load_ohlcv
from .metrics import rmse
from .model import build_lstm
from .preprocessing import inverse_return, prepare_datasets, reconstruct_price
from .utils import configure_logging, set_global_seed

logger = logging.getLogger(__name__)


def _level_based_score(config, frame):
    from tensorflow import keras

    close = frame[["Close"]].astype(float).values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    window = config.preprocessing.window_size
    n = len(scaled)
    split = int(np.floor(n * config.preprocessing.train_fraction))

    x, y, ends = [], [], []
    for i in range(window, n):
        x.append(scaled[i - window:i, 0])
        y.append(scaled[i, 0])
        ends.append(i)
    x, y, ends = np.array(x), np.array(y), np.array(ends)
    is_test = ends >= split

    model = keras.Sequential([
        keras.Input(shape=(window, 1)),
        keras.layers.LSTM(16),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        x[~is_test], y[~is_test],
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        verbose=0,
    )

    pred_scaled = model.predict(x[is_test], verbose=0)
    pred_prices = scaler.inverse_transform(pred_scaled)
    actual_prices = close[ends[is_test], 0]
    return float(rmse(actual_prices, pred_prices.flatten()))


def _return_based_score(config, frame):
    from tensorflow import keras

    pre = config.preprocessing
    data = prepare_datasets(
        frame, pre.features, pre.target, pre.window_size, pre.train_fraction
    )

    val_count = max(1, int(len(data.x_train) * config.training.val_fraction))
    x_fit, y_fit = data.x_train[:-val_count], data.y_train[:-val_count]
    x_val, y_val = data.x_train[-val_count:], data.y_train[-val_count:]

    model = build_lstm(
        pre.window_size,
        len(pre.features),
        config.model.lstm_units,
        config.model.dropout,
        config.model.learning_rate,
    )
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.training.early_stopping_patience,
        restore_best_weights=True,
    )
    model.fit(
        x_fit, y_fit,
        validation_data=(x_val, y_val),
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        callbacks=[stopper],
        verbose=0,
    )

    scaled_pred = model.predict(data.x_test, verbose=0)
    predicted_return = inverse_return(scaled_pred, data.scaler, data.target_index)
    predicted_price = reconstruct_price(data.prev_close_test, predicted_return)
    return float(rmse(data.actual_close_test, predicted_price))


def run(config):
    set_global_seed(config.seed)
    frame = load_ohlcv(config.data.raw_csv)

    level_rmse = _level_based_score(config, frame)
    set_global_seed(config.seed)
    return_rmse = _return_based_score(config, frame)

    reduction = (level_rmse - return_rmse) / level_rmse * 100
    summary = {
        "level_based_lstm_rmse": level_rmse,
        "return_based_stacked_lstm_rmse": return_rmse,
        "rmse_reduction_pct": reduction,
    }

    config.artifacts.results_dir.mkdir(parents=True, exist_ok=True)
    (config.artifacts.results_dir / "ablation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info(
        "Level-based RMSE %.2f -> Return-based RMSE %.2f | Reduction %.1f%%",
        level_rmse,
        return_rmse,
        reduction,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: level-based LSTM vs return-based stacked LSTM."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    configure_logging()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
