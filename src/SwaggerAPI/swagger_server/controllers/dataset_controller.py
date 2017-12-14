import connexion
import six

from swagger_server.models.api_response import ApiResponse  # noqa: E501
from swagger_server.models.inline_image import InlineImage  # noqa: E501
from swagger_server.models.inline_response200 import InlineResponse200  # noqa: E501
from swagger_server.models.table import Table  # noqa: E501
from swagger_server import util

import sys
import os

sys.path.append(os.path.abspath("C:/Users/Matt/Documents/GitHub/mscs710-project/src"))
from utils import *

import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from matplotlib import style
from subprocess import check_output

import models
import utils
import visuals

warnings.filterwarnings("ignore")
style.use('ggplot')
plt.rcParams['figure.figsize'] = (12.0, 8.0)

#Globals
currencies = visuals.load()
coin_type = 'bitcoin'
coin_feat = ['Close']
fileName = 'bitcoin_price.csv'
input_dir = "../input"
df = utils.load_data(input_dir, fileName)
graphType = 1
corr_choice = 1
model_type = 1
operation_type = 1
model, x_train, y_train, x_test, y_test, x_recent = utils.load_data(input_dir, fileName)

def dataset_currency_selection_currency_name_post(currencyName):  # noqa: E501
    """Selects a specific currency type

    The currency type should be either bitcoin, bitconnect, dash, ethereum, iota, litecoin, monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, or waves # noqa: E501

    :param currencyName: Should be equal to either bitcoin, bitconnect, dash, ethereum, iota, litecoin, monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, or waves
    :type currencyName: str

    :rtype: ApiResponse
    """
    #print('Enter Currency File Name: ')
    # name = input()
    global fileName
    fileName = currencyName + '_price.csv'
    global input_dir

    global x_train, x_test, x_recent, y_train, y_test, df
    x_train, x_test, x_recent, y_train, y_test, df = utils.load_data(input_dir, fileName)

    print('Building Summary...')

    summary = ''
    summary += 'Date of newest data: {}'.format(df.index[0])
    summary += 'Date of oldest data: {}\n'.format(df.index[-1])
    summary += str(x_train.shape[0]) + 'training samples.'
    summary += str(x_test.shape[0]) + 'test samples.'
    summary += 'Predicting {} days'.format(x_recent.shape[0])
    summary += 'Train sample shape: ' + str(x_train.shape)
    summary += 'Test sample shape: ' + str(x_test.shape)
    summary += 'Train label shape:' + str(y_train.shape)
    summary += 'Test label shape:' + str(y_test.shape)
    summary += 'Sample Data: '
    summary += str(df.describe())
    print('Successfully executed method')
    return summary


def dataset_get_stats_get():  # noqa: E501
    """Get Statistics for the Dataset

    Returns values of count, mean, median, mode, min, max, standard deviation (std) for the dataset # noqa: E501


    :rtype: InlineResponse200
    """
    return 'do some magic!'


def dataset_get_table_table_name_post(tableName):  # noqa: E501
    """Returns the specified table

    Returns the HTML Representation of the table. # noqa: E501

    :param tableName: Should be equal to either bitcoin, bitconnect, dash, ethereum, iota, litecoin, monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, or waves
    :type tableName: str

    :rtype: Table
    """

    print(currencies[tableName].head())
    return str(currencies[tableName].head())


def dataset_graph_correlation_correlation_type_post(correlationType):  # noqa: E501
    """Returns a graph utilizing the selected correlation type

    1. Spearman, 2. Pearson, 3. Kendall # noqa: E501

    :param correlationType: Must be a number equal to 1, 2, or 3
    :type correlationType: int

    :rtype: InlineImage
    """

    global corr_choice
    corr_choice = correlationType
    if corr_choice == 1:
        return visuals.plot_correlation('spearman')
    elif corr_choice == 2:
        return visuals.plot_correlation('pearson')
    elif corr_choice == 3:
        return visuals.plot_correlation('kendall')
    return 'Error: The given number was not 1, 2, or 3'


def dataset_graph_selection_graph_selection_post(graphSelection):  # noqa: E501
    """Selects a graph type

    1. Trend Curve, 2. Candlestick, 3. Correlation Map # noqa: E501

    :param graphSelection: Must be a number equal to 1, 2, or 3
    :type graphSelection: int

    :rtype: ApiResponse
    """
    global graphType
    graphType = graphSelection
    if graphType == 1:
        visuals.plot_trend(currencies, coin_type, coin_feat)
    elif graphType == 2:
        visuals.plot_candlestick(currencies, coin_type, coin_feat)
    elif graphType == 3:
        if corr_choice == 1:
            return visuals.plot_correlation('spearman')
        elif corr_choice == 2:
            return visuals.plot_correlation('pearson')
        elif corr_choice == 3:
            return visuals.plot_correlation('kendall')
    return 'Error'


def dataset_model_type_model_type_post(modelType):  # noqa: E501
    """Selects a model type

    1. Linear Regression, 2. Support Vector Regression, 3. Multilayer Perceptron, 4. Gradient Boosting Regression # noqa: E501

    :param modelType: Must be a number equal to 1, 2, 3, or 4
    :type modelType: int

    :rtype: ApiResponse
    """
    global model_Type
    model_Type = modelType
    global model

    if model_Type == 1:
        model = models.LR(x_train, y_train, x_test, y_test, x_recent)
        return print('Linear Regression model selected.\n')
    elif model_Type == 2:
        model = models.SVR(x_train, y_train, x_test, y_test, x_recent)
        return print('Support Vector Regression model selected.\n')
    elif model_Type == 3:
        model = models.MLP(x_train, y_train, x_test, y_test, x_recent)
        model.build()
        return print('Multilayer Perceptron model selected.\n')
    elif model_Type == 4:
        model = models.GTBR(x_train, y_train, x_test, y_test, x_recent)
        return print('Gradient Boosting Regression model selected.\n')
    return 'Error'


def dataset_operation_type_operation_type_post(operationType):  # noqa: E501
    """Selects what operation to do, either training, testing, or prediction

    1. Train, 2. Test, 3. Predict # noqa: E501

    :param operationType: Must be a number equal to 1, 2, or 3
    :type operationType: int

    :rtype: ApiResponse
    """
    global operation_type
    operation_type = operationType

    if operation_type == 1:
        print('Training initiated...\n')
        model.train()
        return
    elif operation_type == 2:
        print('Evaluating model on test data...\n')
        model.test()
        return
    elif operation_type == 3:
        print('Predicting future values...\n')
        preds = model.predict()
        print('Forecast Plot')
        return utils.forecast_plot(df, preds)
    return 'Error'


def dataset_train_currency_name_post(currencyName):  # noqa: E501
    """Trains based on the currency selected

    Should be either bitcoin, bitconnect, dash, ethereum, iota, litecoin, monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, or waves # noqa: E501

    :param currencyName: Should be equal to either bitcoin, bitconnect, dash, ethereum, iota, litecoin, monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, or waves
    :type currencyName: str

    :rtype: ApiResponse
    """
    #print('Enter Currency File Name: ')
    # name = input()
    global fileName
    fileName = currencyName+"_price.csv"
    global x_train, x_test, x_recent, y_train, y_test, df

    x_train, x_test, x_recent, y_train, y_test, df = utils.load_data(input_dir, fileName)
    print('---------------------------------------')
    print(x_train.shape[0], 'training samples.')
    print(x_test.shape[0], 'test samples.')
    print('Predicting {} days'.format(x_recent.shape[0]))
    print('Train sample shape: ', x_train.shape)
    print('Test sample shape: ', x_test.shape)
    print('Train label shape:', y_train.shape)
    print('Test label shape:', y_test.shape)
    return 'Trained for currency'
