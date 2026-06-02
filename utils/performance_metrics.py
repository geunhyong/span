
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """
    모델 성능을 평가하는 함수
    :param y_true: 실제 값
    :param y_pred: 예측 값
    :return: RMSE 및 R²
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2
