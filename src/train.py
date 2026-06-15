from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from .config import load_config
from .data import load_ohlcv
from .model import build_lstm
from .preprocessing import prepare_datasets
from .utils import configure_logging, set_global_seed
from .visualize import plot_training_history

logger = logging.getLogger(__name__)


def train(config):
    import joblib

    set_global_seed(config.seed)
    frame = load_ohlcv(config.data.raw_csv)
    pre = config.preprocessing

    data = prepare_datasets(
        frame, pre.features, pre.target, pre.window_size, pre.train_fraction
    )
    logger.info("Train windows: %d | Test windows: %d", len(data.x_train), len(data.x_test))

    val_count = max(1, int(len(data.x_train) * config.training.val_fraction))
    x_fit, y_fit = data.x_train[:-val_count], data.y_train[:-val_count]
    x_val, y_val = data.x_train[-val_count:], data.y_train[-val_count:]

    from tensorflow import keras

    model = build_lstm(
        pre.window_size,
        len(pre.features),
        config.model.lstm_units,
        config.model.dropout,
        config.model.learning_rate,
    )
    model.summary(print_fn=logger.info)

    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.training.early_stopping_patience,
        restore_best_weights=True,
    )
    history = model.fit(
        x_fit,
        y_fit,
        validation_data=(x_val, y_val),
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        callbacks=[stopper],
        verbose=2,
    )

    config.artifacts.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(config.artifacts.model_path)
    joblib.dump(data.scaler, config.artifacts.scaler_path)

    figures = config.artifacts.results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_training_history(history.history, figures / "training_loss.png")
    logger.info("Saved model to %s", config.artifacts.model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the stacked LSTM.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    configure_logging()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
