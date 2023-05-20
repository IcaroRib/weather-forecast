import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    mean_squared_error, mean_absolute_percentage_error, r2_score



def describe_dist(df, feature):
    media = df[feature].mean()
    mediana = df[feature].median()
    desvio = df[feature].std()
    assimetria = df[feature].skew()
    curtose = df[feature].kurtosis()
    min = df[feature].min()
    max = df[feature].max()

    print("Média          ", media)
    print("Mediana        ", mediana)
    print("Desvio Padrão  ", desvio)
    print("Assimetria     ", assimetria)
    print("Curtose        ", curtose)
    print("Mínimo         ", min)
    print("Máximo         ", max)


def hypothesis_result(stat, p_value, alpha):
    print(f'Statistics = {stat:.10f}')
    print(f'P-value    = {p_value:.16f}')

    if p_value > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

    plt.rcParams['figure.figsize'] = [10, 5]


def read_dataset(path, target_name="temp"):
    columns = ['date', 'hour', 'prep_tot',
               'pres_atm', 'max_pres_atm', 'min_pres_atm',
               "rad_glob",
               "temp", "dew_point", "max_temp", "min_temp",
               "max_dew", "min_dew",
               "max_humi", "min_humi", "humi",
               "wind_direc", "max_wind", "wind_speed", "unnamed"]

    combined_df = pd.DataFrame()
    for filename in os.listdir(path):
        fullpath = os.path.join(path, filename)
        df = pd.read_csv(fullpath, names=columns, header=8, encoding='latin-1', sep=';', dtype=str)

        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['hour'] = df['hour'].str[:2]
        df['hour'] = df['hour'].astype(int)
        for column in columns[2:]:
            df[column] = df[column].str.replace(",", ".")
            df[column] = df[column].astype(float)

        combined_df = pd.concat([combined_df, df], ignore_index=True)

    final_df = combined_df.drop("unnamed", axis=1)
    return final_df


def clean_df(df):
    df = df[(df != -9999).all(axis=1)]
    return df


def resample_df(df, metrics):
    daily_df = df[metrics].loc[(df['prep_tot'].notnull()) & (df['temp'].notnull())].copy()
    daily_df = daily_df.loc[(daily_df['prep_tot'] != -9999) & (daily_df['temp'] != -9999)]
    daily_df['fulldate'] = df['date'] + " " + df['hour'].astype(str) + ":00:00"
    daily_df['fulldate'] = pd.to_datetime(daily_df['fulldate'])
    daily_df = daily_df[['fulldate'] + metrics]

    rule = {}
    for metric in metrics:
        if metric == 'prep_tot':
            rule[metric] = 'sum'
        else:
            rule[metric] = "mean"

    daily_df = daily_df.resample('1D', on='fulldate').agg(rule).reset_index()
    daily_df.set_index('fulldate', inplace=True)
    return daily_df



def get_score(y_true, y_pred):
  # Calculate accuracy
  accuracy = accuracy_score(y_true, y_pred)

  # Calculate precision
  precision = precision_score(y_true, y_pred)

  # Calculate recall
  recall = recall_score(y_true, y_pred)

  # Calculate F1-score
  f1_ = f1_score(y_true, y_pred)

  return {
      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "f1-score": f1_
  }