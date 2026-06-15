from __future__ import annotations

import argparse
import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .config import OHLCV_COLUMNS, load_config

logger = logging.getLogger(__name__)


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[list(OHLCV_COLUMNS)].copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.dropna()
    return frame


def download_ohlcv(ticker: str, lookback_days: int, end_date=None, out_path=None) -> pd.DataFrame:
    import yfinance as yf

    end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.today().normalize()
    start = end - timedelta(days=lookback_days)
    history = yf.Ticker(ticker).history(
        start=start.date().isoformat(),
        end=(end + timedelta(days=1)).date().isoformat(),
        interval="1d",
        auto_adjust=True,
    )
    if history.empty:
        raise RuntimeError(
            f"No data returned for '{ticker}'. Check the ticker symbol and network access."
        )
    frame = _clean(history)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index_label="Date")
        logger.info("Saved %d rows to %s", len(frame), out_path)
    return frame


def load_ohlcv(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run `python -m src.data --config config.yaml` first."
        )
    frame = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    return _clean(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily OHLCV data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    download_ohlcv(
        config.data.ticker,
        config.data.lookback_days,
        config.data.end_date,
        config.data.raw_csv,
    )


if __name__ == "__main__":
    main()
