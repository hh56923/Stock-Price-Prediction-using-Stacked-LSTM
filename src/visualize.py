from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_training_history(history, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["loss"], label="train")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled returns)")
    ax.set_title("Training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_predictions(dates, actual, predicted, out_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, actual, label="Actual", linewidth=1.6)
    ax.plot(dates, predicted, label="Predicted", linewidth=1.6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close price (INR)")
    ax.set_title("Maruti close: actual vs predicted (test set)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
