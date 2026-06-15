from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class DataConfig:
    ticker: str
    lookback_days: int
    end_date: date | None
    raw_csv: Path


@dataclass(frozen=True)
class PreprocessingConfig:
    features: tuple[str, ...]
    target: str
    window_size: int
    train_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    lstm_units: tuple[int, ...]
    dropout: float
    learning_rate: float


@dataclass(frozen=True)
class BaselineConfig:
    lstm_units: tuple[int, ...]
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    val_fraction: float
    early_stopping_patience: int


@dataclass(frozen=True)
class ArtifactsConfig:
    model_path: Path
    scaler_path: Path
    results_dir: Path


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    baseline: BaselineConfig
    training: TrainingConfig
    artifacts: ArtifactsConfig


def _parse_end_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        config = Config(
            seed=int(raw["seed"]),
            data=DataConfig(
                ticker=str(raw["data"]["ticker"]),
                lookback_days=int(raw["data"]["lookback_days"]),
                end_date=_parse_end_date(raw["data"]["end_date"]),
                raw_csv=Path(raw["data"]["raw_csv"]),
            ),
            preprocessing=PreprocessingConfig(
                features=tuple(raw["preprocessing"]["features"]),
                target=str(raw["preprocessing"]["target"]),
                window_size=int(raw["preprocessing"]["window_size"]),
                train_fraction=float(raw["preprocessing"]["train_fraction"]),
            ),
            model=ModelConfig(
                lstm_units=tuple(int(u) for u in raw["model"]["lstm_units"]),
                dropout=float(raw["model"]["dropout"]),
                learning_rate=float(raw["model"]["learning_rate"]),
            ),
            baseline=BaselineConfig(
                lstm_units=tuple(int(u) for u in raw["baseline"]["lstm_units"]),
                dropout=float(raw["baseline"]["dropout"]),
            ),
            training=TrainingConfig(
                epochs=int(raw["training"]["epochs"]),
                batch_size=int(raw["training"]["batch_size"]),
                val_fraction=float(raw["training"]["val_fraction"]),
                early_stopping_patience=int(raw["training"]["early_stopping_patience"]),
            ),
            artifacts=ArtifactsConfig(
                model_path=Path(raw["artifacts"]["model_path"]),
                scaler_path=Path(raw["artifacts"]["scaler_path"]),
                results_dir=Path(raw["artifacts"]["results_dir"]),
            ),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Config file '{path}' is missing or malformed at key: {exc}") from exc
    _validate(config)
    return config


def _validate(config: Config) -> None:
    pre, model, train = config.preprocessing, config.model, config.training

    unknown = [f for f in pre.features if f not in OHLCV_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown feature column(s) {unknown}; choose from {list(OHLCV_COLUMNS)}.")
    if pre.target not in pre.features:
        raise ValueError(
            f"Target '{pre.target}' must be listed in preprocessing.features."
        )
    if pre.window_size < 2:
        raise ValueError("preprocessing.window_size must be at least 2.")
    if not 0.0 < pre.train_fraction < 1.0:
        raise ValueError("preprocessing.train_fraction must be strictly between 0 and 1.")

    if not model.lstm_units or any(u < 1 for u in model.lstm_units):
        raise ValueError("model.lstm_units must be a non-empty list of positive integers.")
    if not 0.0 <= model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1).")
    if model.learning_rate <= 0:
        raise ValueError("model.learning_rate must be positive.")

    if not config.baseline.lstm_units or any(u < 1 for u in config.baseline.lstm_units):
        raise ValueError("baseline.lstm_units must be a non-empty list of positive integers.")

    if train.epochs < 1 or train.batch_size < 1:
        raise ValueError("training.epochs and training.batch_size must be positive.")
    if not 0.0 < train.val_fraction < 0.5:
        raise ValueError("training.val_fraction must be in (0, 0.5).")
    if train.early_stopping_patience < 1:
        raise ValueError("training.early_stopping_patience must be positive.")

    if config.data.lookback_days <= pre.window_size:
        raise ValueError("data.lookback_days must exceed preprocessing.window_size.")
