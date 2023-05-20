from src.utils import *
from statsmodels.tsa.seasonal import seasonal_decompose
from sktime.forecasting.base import ForecastingHorizon
import pickle


def read_database(path):
    metrics = ['prep_tot', 'pres_atm', 'max_pres_atm', 'min_pres_atm',
               "rad_glob",
               "temp", "dew_point", "max_temp", "min_temp",
               "max_dew", "min_dew",
               "max_humi", "min_humi", "humi",
               "wind_direc", "max_wind", "wind_speed"]

    df = read_dataset(path)
    df = clean_df(df)
    daily_df = resample_df(df, metrics)

    return daily_df


def decompose_series(ts, model='additive', period=365):
    res = seasonal_decompose(ts.dropna(), model=model, period=period, extrapolate_trend='freq')
    size = len(res.resid)

    residue = pd.Series(data=res.resid[365:size-365], index=ts.index[365:size-365])
    seasonal = pd.Series(data=res.seasonal[365:size-365], index=ts.index[365:size-365])
    trend = pd.Series(data=res.trend[365:size-365], index=ts.index[365:size-365])

    residue = residue.fillna(method='ffill')
    residue = residue.fillna(method='bfill')

    seasonal = seasonal.fillna(method='ffill')
    seasonal = seasonal.fillna(method='bfill')

    trend = trend.fillna(method='ffill')
    trend = trend.fillna(method='bfill')

    return residue, seasonal, trend


def recompose_series(y, trend, seasonal):
    start_date = y.index.min()
    end_date = y.index.max()
    y_trend = trend.loc[(trend.index >= start_date) & (trend.index <= end_date)]
    y_seasonal = seasonal.loc[(seasonal.index >= start_date) & (seasonal.index <= end_date)]

    return pd.Series(data=y.values + y_trend.values + y_seasonal.values, index=y.index)


def get_window(y_train, y_test):
    fh_abs = ForecastingHorizon(y_test.index, is_relative=False, freq="D")

    cutoff = y_train.index[-1]
    fh_rel = fh_abs.to_relative(cutoff)

    cutoff_insample = y_test.index[-1]
    fh_rel_insample = fh_abs.to_relative(cutoff_insample)

    return fh_rel, fh_rel_insample


def save_model(path, res):

    model = res.pop("model")
    filename = res.pop("filename")
    y_test = res.pop("y_test")
    #y_pred = res.pop("y_pred")

    index = [res.pop('name')]
    params = res.pop("parameters")
    if params:
        for k, v in params.items():
            if type(v) == list or type(v) == tuple:
                params[k] = str(v)
        res.update(params)

    df = pd.DataFrame(res, index=index)
    df.to_csv(f"{path}/results.csv")

    df = pd.DataFrame(y_test)
    df.to_csv(f"{path}/y_test.csv")

    #df = pd.DataFrame(y_pred)
    #df.to_csv(f"{path}/y_pred.csv")

    if model:
        with open(f'{path}/{filename}.pkl', 'wb') as f:
            pickle.dump(model, f)


def save_failed(path, params):
    if not os.path.exists(path):
        os.mkdir(path)

    with open(f'{path}/error_file.txt', 'a') as f:
        text = ';'.join(params)
        f.write(text+"\n")