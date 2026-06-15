from __future__ import annotations

import numpy as np


def rmse(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def r2(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot else 0.0


def directional_accuracy(prev_close, actual, predicted):
    prev_close = np.asarray(prev_close)
    actual_dir = np.sign(np.asarray(actual) - prev_close)
    pred_dir = np.sign(np.asarray(predicted) - prev_close)
    return float(np.mean(actual_dir == pred_dir) * 100)
