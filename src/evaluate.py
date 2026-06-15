from __future__ import annotations

import argparse
import json
import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd

from .config import load_config
from .data import load_ohlcv
from .metrics import directional_accuracy, mae, mape, r2, rmse
from .preprocessing import inverse_return, prepare_datasets, reconstruct_price
from .utils import configure_logging
from .visualize import plot_predictions

logger = logging.getLogger(__name__)


def evaluate(config):
    import joblib
    from tensorflow import keras

    if not config.artifacts.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at '{config.artifacts.model_path}'. Run `python -m src.train` first."
        )
    if not config.artifacts.scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at '{config.artifacts.scaler_path}'. Run `python -m src.train` first."
        )

    frame = load_ohlcv(config.data.raw_csv)
    pre = config.preprocessing
    scaler = joblib.load(config.artifacts.scaler_path)
    data = prepare_datasets(
        frame, pre.features, pre.target, pre.window_size, pre.train_fraction, scaler=scaler
    )

    model = keras.models.load_model(config.artifacts.model_path)
    scaled_pred = model.predict(data.x_test, verbose=0)
    predicted_return = inverse_return(scaled_pred, scaler, data.target_index)

    predicted_price = reconstruct_price(data.prev_close_test, predicted_return)
    naive_price = data.prev_close_test
    actual_price = data.actual_close_test

    metrics = {
        "model": {
            "rmse": rmse(actual_price, predicted_price),
            "mae": mae(actual_price, predicted_price),
            "mape": mape(actual_price, predicted_price),
            "r2": r2(actual_price, predicted_price),
            "directional_accuracy": directional_accuracy(
                data.prev_close_test, actual_price, predicted_price
            ),
        },
        "naive_persistence": {
            "rmse": rmse(actual_price, naive_price),
            "mae": mae(actual_price, naive_price),
            "mape": mape(actual_price, naive_price),
        },
        "test_size": int(len(actual_price)),
    }

    config.artifacts.results_dir.mkdir(parents=True, exist_ok=True)
    (config.artifacts.results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    predictions = pd.DataFrame(
        {
            "Date": data.test_dates,
            "actual_close": actual_price,
            "predicted_close": predicted_price,
            "naive_last_value": naive_price,
        }
    )
    predictions.to_csv(config.artifacts.results_dir / "predictions.csv", index=False)

    figures = config.artifacts.results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_predictions(data.test_dates, actual_price, predicted_price, figures / "predictions.png")

    logger.info(
        "Model RMSE %.2f | Naive RMSE %.2f | MAPE %.2f%% | Dir. acc %.1f%%",
        metrics["model"]["rmse"],
        metrics["naive_persistence"]["rmse"],
        metrics["model"]["mape"],
        metrics["model"]["directional_accuracy"],
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    configure_logging()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
