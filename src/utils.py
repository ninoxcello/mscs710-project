import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation


def load_data(input_dir, file_name, forecast_col='Close'):
    # read in csv
    df = pd.read_csv('{}/{}'.format(input_dir, file_name), parse_dates=['Date'], index_col=0)
    # select & add feature columns
    df.fillna(0, inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']]
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.
    df['PCT_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.
    df = df.iloc[::-1]
    df.fillna(value=-9999, inplace=True)
    # set # of days to forecast out and shift column to be used as labels
    days_forecast = 15
    df['label'] = df[forecast_col].shift(-days_forecast)
    # set up feature & label matrices
    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-days_forecast:]
    X = X[:-days_forecast]
    df.dropna(inplace=True)
    y = np.array(df['label'])
    # split data 80/20 for train & test respectively
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    return X_train, X_test, X_lately, y_train, y_test, df


def forecast_plot(df, predictions):
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400  # sec in day
    next_unix = last_unix + one_day

    for i in predictions:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    plt.plot(df['Close'])
    plt.plot(df['Forecast'])
    plt.legend(bbox_to_anchor=(1.01, 1))
    plt.xlabel('Time(Yr-M)')
    plt.ylabel('Value(USD)')
    plt.show()