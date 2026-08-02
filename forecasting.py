"""Forecasting utilities for AIDCE-Sim.

Models are deliberately interpretable and lightweight:
1) seasonal naive,
2) Holt-Winters exponential smoothing,
3) gradient boosting with lag and calendar features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class ForecastResult:
    metrics: pd.DataFrame
    holdout: pd.DataFrame
    future: pd.DataFrame
    best_model: str


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.clip(np.abs(y_true), 1e-6, None)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    return {"MAE (MW)": mae, "RMSE (MW)": rmse, "MAPE (%)": mape}


def _seasonal_naive(history: np.ndarray, steps: int, season: int) -> np.ndarray:
    if len(history) == 0:
        return np.zeros(steps)
    season = max(1, min(season, len(history)))
    template = history[-season:]
    return np.resize(template, steps).astype(float)


def _holt_winters(history: np.ndarray, steps: int, season: int) -> np.ndarray:
    season = max(2, season)
    if len(history) < 2 * season:
        raise ValueError("At least two seasonal cycles are required for Holt-Winters.")
    model = ExponentialSmoothing(
        history,
        trend="add",
        seasonal="add",
        seasonal_periods=season,
        initialization_method="estimated",
    ).fit(optimized=True, use_brute=False)
    return np.asarray(model.forecast(steps), dtype=float)


def _feature_vector(
    history: List[float],
    timestamp: pd.Timestamp,
    origin: pd.Timestamp,
    freq_minutes: int,
    season: int,
) -> List[float]:
    def lag(k: int) -> float:
        return history[-k] if len(history) >= k else history[0]

    roll_short_n = max(2, int(round(60 / freq_minutes)))
    roll_long_n = max(roll_short_n, min(season, len(history)))
    hour = timestamp.hour + timestamp.minute / 60.0
    dow = timestamp.dayofweek
    trend = (timestamp - origin).total_seconds() / 3600.0
    return [
        lag(1),
        lag(2),
        lag(max(3, roll_short_n)),
        lag(min(season, len(history))),
        float(np.mean(history[-roll_short_n:])),
        float(np.mean(history[-roll_long_n:])),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        trend,
    ]


def _fit_gradient_boosting(
    series: pd.Series, freq_minutes: int, season: int
) -> HistGradientBoostingRegressor:
    values = series.to_numpy(dtype=float)
    max_lag = max(season, int(round(60 / freq_minutes)), 4)
    if len(values) <= max_lag + 20:
        raise ValueError("Insufficient history for gradient boosting.")
    X, y = [], []
    origin = series.index[0]
    history = list(values[:max_lag])
    for i in range(max_lag, len(values)):
        X.append(_feature_vector(history, series.index[i], origin, freq_minutes, season))
        y.append(values[i])
        history.append(values[i])
    model = HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.055,
        max_leaf_nodes=24,
        l2_regularization=0.15,
        random_state=42,
    )
    model.fit(np.asarray(X), np.asarray(y))
    return model


def _recursive_gradient_forecast(
    model: HistGradientBoostingRegressor,
    history_series: pd.Series,
    future_index: pd.DatetimeIndex,
    freq_minutes: int,
    season: int,
) -> np.ndarray:
    history = list(history_series.to_numpy(dtype=float))
    origin = history_series.index[0]
    predictions = []
    for ts in future_index:
        x = np.asarray(
            [_feature_vector(history, ts, origin, freq_minutes, season)], dtype=float
        )
        pred = max(0.0, float(model.predict(x)[0]))
        predictions.append(pred)
        history.append(pred)
    return np.asarray(predictions)


def run_forecasting(
    series: pd.Series,
    freq_minutes: int,
    forecast_hours: int = 24,
    holdout_hours: int = 24,
) -> ForecastResult:
    series = series.dropna().astype(float).sort_index()
    if len(series) < 48:
        raise ValueError("At least 48 observations are required for forecasting.")

    season = max(2, int(round(24 * 60 / freq_minutes)))
    holdout_steps = int(round(holdout_hours * 60 / freq_minutes))
    forecast_steps = int(round(forecast_hours * 60 / freq_minutes))
    holdout_steps = max(4, min(holdout_steps, max(4, len(series) // 4)))
    forecast_steps = max(1, forecast_steps)

    train = series.iloc[:-holdout_steps]
    test = series.iloc[-holdout_steps:]
    candidates: Dict[str, np.ndarray] = {}

    candidates["Seasonal naive"] = _seasonal_naive(
        train.to_numpy(), holdout_steps, season
    )
    try:
        candidates["Holt-Winters"] = np.clip(
            _holt_winters(train.to_numpy(), holdout_steps, season), 0.0, None
        )
    except Exception:
        pass
    try:
        gb = _fit_gradient_boosting(train, freq_minutes, season)
        candidates["Gradient boosting"] = _recursive_gradient_forecast(
            gb, train, test.index, freq_minutes, season
        )
    except Exception:
        pass

    metric_rows = []
    for name, pred in candidates.items():
        row = {"Model": name, **_metrics(test.to_numpy(), pred)}
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows).sort_values("RMSE (MW)").reset_index(drop=True)
    best_model = str(metrics.iloc[0]["Model"])

    holdout = pd.DataFrame({"actual_mw": test.to_numpy()}, index=test.index)
    for name, pred in candidates.items():
        holdout[name] = pred

    future_index = pd.date_range(
        start=series.index[-1] + pd.Timedelta(minutes=freq_minutes),
        periods=forecast_steps,
        freq=f"{freq_minutes}min",
    )
    if best_model == "Seasonal naive":
        point = _seasonal_naive(series.to_numpy(), forecast_steps, season)
    elif best_model == "Holt-Winters":
        point = np.clip(
            _holt_winters(series.to_numpy(), forecast_steps, season), 0.0, None
        )
    else:
        gb_full = _fit_gradient_boosting(series, freq_minutes, season)
        point = _recursive_gradient_forecast(
            gb_full, series, future_index, freq_minutes, season
        )

    residuals = holdout["actual_mw"].to_numpy() - holdout[best_model].to_numpy()
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    horizon_scale = np.sqrt(1.0 + np.arange(1, forecast_steps + 1) / max(season, 1))
    delta = 1.96 * sigma * horizon_scale
    future = pd.DataFrame(
        {
            "forecast_mw": point,
            "lower_95_mw": np.clip(point - delta, 0.0, None),
            "upper_95_mw": point + delta,
        },
        index=future_index,
    )
    return ForecastResult(metrics=metrics, holdout=holdout, future=future, best_model=best_model)
