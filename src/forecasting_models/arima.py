import numpy as np
import pandas as pd
import itertools as itert

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.forecasting_models.forecasting import decompose_series, get_window, recompose_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import ARIMA


def __forecast(model, y_train, y_test, trend, seasonal):
    fh_rel, fh_rel_insample = get_window(y_train, y_test)

    model.fit(y_train)

    y_pred_rel = model.predict(fh=fh_rel)

    true_y_test = recompose_series(y_test, trend, seasonal)
    true_y_pred_rel = recompose_series(y_pred_rel, trend, seasonal)

    try:
        rmse = mean_squared_error(y_true=true_y_test, y_pred=true_y_pred_rel, squared=False)
    except:
        print(trend.values)
        print(seasonal.values)
        print(y_test.values)
        print(y_pred_rel.values)
        print(true_y_test.values)
    mape = mean_absolute_percentage_error(y_true=true_y_test, y_pred=true_y_pred_rel)
    if len(true_y_test) > 2:
        r_2 = r2_score(y_true=true_y_test, y_pred=true_y_pred_rel)
    else:
        r_2 = 0

    return rmse, mape, r_2


def __evaluate(y, trend, seasonal, splitter, params):

    rmse_list = []
    mape_list = []
    r2_list = []

    model = ARIMA(**params)

    for j, (train_index, val_index) in enumerate(splitter.split(y)):
        y_train = y[train_index]
        y_val = y[val_index]
        rmse, mape, r_2 = __forecast(model, y_train, y_val, trend, seasonal)
        rmse_list.append(rmse)
        mape_list.append(mape)
        r2_list.append(r_2)

    return np.array(rmse_list).mean(), np.array(mape).mean(), np.array(r2_list).mean()


def __cross_validation(y, trend, seasonal, params_list, steps=5, splits=10):
    splitter = TimeSeriesSplit(gap=round(365/splits), max_train_size=365, n_splits=splits, test_size=steps)
    results = pd.DataFrame(columns=["params", "rmse", "mape", "r2"])

    for i, params in enumerate(params_list):
        rmse, mape, r_2 = __evaluate(y, trend, seasonal, splitter, params)
        results.loc[i] = [params, rmse, mape, r_2]

    return results


def arima_forecast(df, attribute, steps=5):

    print("Starting ARIMA Forecasting... ")

    residue, seasonal, trend = decompose_series(df[attribute])
    y_train, y_test = temporal_train_test_split(residue, test_size=730)

    parameters = []
    for p, d, q in list(itert.product("01234", repeat=3)):
        p, d, q = int(p), int(d), int(q)
        if (p + d + q > 0) and (p + d + q < 5):
            dic = {"order": (p, d, q)}
            parameters.append(dic)

    models = __cross_validation(y_train, trend, seasonal, params_list=parameters, steps=steps)

    best_model_index = models['rmse'].idxmin()
    params = models.loc[best_model_index].params

    print("Best Model found")
    print(models.loc[best_model_index])

    splitter = TimeSeriesSplit(gap=0, max_train_size=365, n_splits=365, test_size=steps)
    rmse, mape, r_2 = __evaluate(y_test, trend, seasonal, splitter, params)

    true_y_test = recompose_series(y_test, trend, seasonal)

    res = {
        "name": "ARIMA",
        "filename": f"arima_{attribute}",
        "attribute": attribute,
        "model": ARIMA(**params),
        "steps": steps,
        "parameters": params,
        "y_test": true_y_test,
        #"y_pred": true_y_pred_rel,
        "rmse": rmse,
        "mape": mape,
        "r2": r_2
    }

    return res