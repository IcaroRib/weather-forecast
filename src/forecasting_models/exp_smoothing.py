import numpy as np
import itertools as itert

from sklearn.model_selection import TimeSeriesSplit
from src.forecasting_models.forecasting import *
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.exp_smoothing import ExponentialSmoothing


def __forecast(model, y_train, y_test):
    fh_rel, fh_rel_insample = get_window(y_train, y_test)

    model.fit(y_train)
    y_pred_rel = model.predict(fh=fh_rel)

    rmse = mean_squared_error(y_true=y_test, y_pred=y_pred_rel, squared=False)
    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred_rel)

    if len(y_pred_rel) > 2:
        r_2 = r2_score(y_true=y_test, y_pred=y_pred_rel)
    else:
        r_2 = 0

    return rmse, mape, r_2


def __evaluate(y, splitter, params):

    rmse_list = []
    mape_list = []
    r2_list = []

    model = ExponentialSmoothing(**params)

    for j, (train_index, val_index) in enumerate(splitter.split(y)):
        y_train = y[train_index]
        y_val = y[val_index]
        rmse, mape, r_2 = __forecast(model, y_train, y_val)
        rmse_list.append(rmse)
        mape_list.append(mape)
        r2_list.append(r_2)

    return np.array(rmse_list).mean(), np.array(mape).mean(), np.array(r2_list).mean()


def __cross_validation(y, params_list, steps=5, splits=10):
    splitter = TimeSeriesSplit(gap=round(365/splits), max_train_size=365, n_splits=splits, test_size=steps)
    results = pd.DataFrame(columns=["params", "rmse", "mape", "r2"])

    for i, params in enumerate(params_list):
        rmse, mape, r_2 = __evaluate(y, splitter, params)
        results.loc[i] = [params, rmse, mape, r_2]

    return results


def exp_smoothing_forecast(df, attribute, steps=5):

    print("Starting Exponential Smoothing Forecast... ")

    size = len(df)
    y = df.iloc[365:size-365][attribute]
    y = y.fillna(method='ffill')

    y_train, y_test = temporal_train_test_split(y, test_size=730)

    parameters = []
    #trends = ["add", "mul"]
    seasonal = ["add", "mul"]
    smoothing_level = [i / 10 for i in range(0, 10, 2)]
    #smoothing_trend = [i / 10 for i in range(0, 10, 3)]
    smoothing_seasonal = [i / 10 for i in range(0, 10, 2)]

    for s, sl, ss in list(itert.product(seasonal, smoothing_level, smoothing_seasonal)):
        dic = {"seasonal": s,
               "smoothing_level": sl,
               "smoothing_seasonal": ss}
        parameters.append(dic)

    models = __cross_validation(y_train, params_list=parameters, steps=steps)

    best_model_index = models['rmse'].idxmin()
    params = models.loc[best_model_index].params

    print("Best Model found")
    print(models.loc[best_model_index])

    splitter = TimeSeriesSplit(gap=0, max_train_size=365, n_splits=365, test_size=steps)
    rmse, mape, r_2 = __evaluate(y_test, splitter, params)

    res = {
        "name": "Exponential Smoothing",
        "filename": f"exp_smoothing_{attribute}",
        "attribute": attribute,
        "model": ExponentialSmoothing(**params),
        "parameters": params,
        "y_test": y_test,
        #"y_pred": y_pred_rel,
        "rmse": rmse,
        "mape": mape,
        "r2": r_2
    }

    return res