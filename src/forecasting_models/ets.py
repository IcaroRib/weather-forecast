import numpy as np
import itertools as itert

from sklearn.model_selection import TimeSeriesSplit
from src.forecasting_models.forecasting import *
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.ets import AutoETS


def __forecast(model, y_train, y_test):
    fh_rel, fh_rel_insample = get_window(y_train, y_test)

    model.fit(y_train)

    y_pred_rel = model.predict(fh=fh_rel)

    rmse = mean_squared_error(y_true=y_test, y_pred=y_pred_rel, squared=False)
    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred_rel)
    r_2 = r2_score(y_true=y_test, y_pred=y_pred_rel)

    return rmse, mape, r_2


def __cross_validation(y, params_list, steps=5, splits=10):
    splitter = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=splits, test_size=steps)
    results = pd.DataFrame(columns=["params", "rmse", "mape", "r2"])

    for i, params in enumerate(params_list):
        rmse_list = []
        mape_list = []
        r2_list = []

        model = AutoETS(**params)

        for j, (train_index, val_index) in enumerate(splitter.split(y)):
            y_train = y[train_index]
            y_val = y[val_index]
            rmse, mape, r_2 = __forecast(model, y_train, y_val)
            rmse_list.append(rmse)
            mape_list.append(mape)
            r2_list.append(r_2)

        results.loc[i] = [params, np.array(rmse_list).mean(), np.array(mape).mean(), np.array(r2_list).mean()]
        #print("Resultados Parciais")
        #print(results)

    return results


def ets_forecast(df, attribute, steps=5):

    size = len(df)
    y = df.iloc[365:size-365][attribute]
    y = y.fillna(method='ffill')

    y_train, y_test = temporal_train_test_split(y, test_size=steps)
    fh_rel, fh_rel_insample = get_window(y_train, y_test)

    parameters = []
    seasonal = ["add", "mul"]
    error = ["add", "mul"]

    for s, e in list(itert.product(seasonal, error)):
        dic = {"seasonal": s,
               "error": e,
               "sp": 365}
        parameters.append(dic)

    models = __cross_validation(y_train, params_list=parameters, steps=5)

    best_model_index = models['rmse'].idxmin()
    params = models.loc[best_model_index].params

    print("Best Model found")
    print(models.loc[best_model_index])

    forecaster = AutoETS(**params)
    forecaster.fit(y_train)

    y_pred_rel = forecaster.predict(fh=fh_rel)

    rmse = mean_squared_error(y_true=y_test, y_pred=y_pred_rel, squared=True)
    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred_rel)
    r_2 = r2_score(y_true=y_test, y_pred=y_pred_rel)
    print(f"rmse = {rmse}")
    print(f"MAPE = {mape}")
    print(f"R2 = {r_2}")

    return rmse, mape, r_2