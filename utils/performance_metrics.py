import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def directional_accuracy(y_true, y_pred) -> float:
    true_sign = np.sign(np.asarray(y_true))
    pred_sign = np.sign(np.asarray(y_pred))
    valid = true_sign != 0
    if valid.sum() == 0:
        return float("nan")
    return float((true_sign[valid] == pred_sign[valid]).mean())


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def evaluate_model(y_true, y_pred) -> tuple[float, float]:
    metrics = evaluate_regression(y_true, y_pred)
    return metrics["rmse"], metrics["r2"]
