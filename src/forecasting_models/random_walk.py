from src.forecasting_models.forecasting import *
from sktime.forecasting.model_selection import temporal_train_test_split
import numpy as np


def __predict(y_test, horizon, steps=5):

    y_pred = []

    for i in range(1, len(horizon)):
        forecast = y_test[i - 1] + np.random.randn()
        y_pred.append(forecast)

    return pd.Series(y_pred, index=horizon[1:])


def random_walk_forecast(df, attribute, steps=5):

    print("Starting Random Walk Forecasting... ")

    size = len(df)
    y = df.iloc[365:size - 365][attribute]
    y = y.fillna(method='ffill')

    y_train, y_test = temporal_train_test_split(y, test_size=366)

    y_pred_rel = __predict(y_test, horizon=y_test.index)

    rmse = mean_squared_error(y_true=y_test[1:], y_pred=y_pred_rel, squared=True)
    mape = mean_absolute_percentage_error(y_true=y_test[1:], y_pred=y_pred_rel)
    r_2 = r2_score(y_true=y_test[1:], y_pred=y_pred_rel)
    print(f"rmse = {rmse}")
    print(f"MAPE = {mape}")
    print(f"R2 = {r_2}")

    res = {
        "name": "Random Walk",
        "filename": f"random_walk_{attribute}",
        "attribute": attribute,
        "model": None,
        "parameters": None,
        "y_test": y_test,
        #"y_pred": y_pred_rel,
        "rmse": rmse,
        "mape": mape,
        "r2": r_2
    }

    return res