from src.forecasting_models.forecasting import read_database, save_model, save_failed
from src.forecasting_models.exp_smoothing import exp_smoothing_forecast
from src.forecasting_models.random_walk import random_walk_forecast
from src.forecasting_models.prophet import prophet_forecast
from src.forecasting_models.arima import arima_forecast
from src.forecasting_models.ets import ets_forecast
from datetime import datetime
import os
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
import pandas as pd
from itertools import product


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=ValueWarning)
    attrs = ['temp', 'prep_tot', 'max_temp', 'min_temp']
    #attrs = ['temp']
    cities = ['curitiba_a807', 'manaus_a101', 'teresina_a312', 'rio_janeiro_a652']
    #steps = [1, 2, 5, 10, 20, 30]
    steps = [1]
    time = datetime.now().strftime('%Y-%m-%d-%H-%M')

    for city, steps, attr in product(cities, steps, attrs):
        path = f'../datasets/{city}'
        ts_df = read_database(path)

        print("Finished reading dataset")
        functions = [arima_forecast, random_walk_forecast, exp_smoothing_forecast, prophet_forecast]
        names = []

        results = {
            "rmse": [],
            "mape": [],
            "r2": [],
        }

        results_path = f'../experiments/{time}/{city}'

        for forecast in functions:

            try:
                res = forecast(ts_df, attr, steps=steps)

                exp_path = f'{results_path}/{res["filename"]}'
                if not os.path.exists(exp_path):
                    os.makedirs(exp_path)

                names.append(res["name"])
                results["rmse"].append(res["rmse"])
                results["mape"].append(res["mape"])
                results["r2"].append(res["r2"])

                save_model(exp_path, res)
            except Exception as e:
                 print(e)
                 params = [city, str(steps), attr, forecast.__name__]
                 save_failed(f'../experiments/{time}', params)

        if len(results['rmse']) > 0:
            df = pd.DataFrame(results, index=names)
            df.to_csv(f"{results_path}/results_{attr}.csv")