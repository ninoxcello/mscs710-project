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
currencies = "UNDEFINED"
coin_type = "UNDEFINED"
coin_feat = "UNDEFINED"
model = "UNDEFINED"
x_train = "UNDEFINED"
y_train = "UNDEFINED"
x_test = "UNDEFINED"
y_test = "UNDEFINED"
x_recent = "UNDEFINED"
name = "UNDEFINED"
input_dir = "UNDEFINED"
df = utils.load_data(input_dir, name)

def dataset_currency_selection_currency_name_post(currencyName):  # noqa: E501
    """Selects a specific currency type

    The currency type should be either bitcoin, bitconnect, dash, ethereum, iota, litecoin, monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, or waves # noqa: E501

    :param currencyName: Should be equal to either bitcoin, bitconnect, dash, ethereum, iota, litecoin, monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, or waves
    :type currencyName: str

    :rtype: ApiResponse
    """
    print('Enter Currency File Name: ')
    # name = input()
    global name
    name = 'bitcoin_price.csv'

    x_train, x_test, x_recent, y_train, y_test, df = utils.load_data(input_dir, name)

    def print_summary(x_train, x_test, x_recent, y_train, y_test, df):
        print('=================================================')
        print('Date of newest data: {}'.format(df.index[0]))
        print('Date of oldest data: {}\n'.format(df.index[-1]))

        # Dataset shapes
        print(x_train.shape[0], 'training samples.')
        print(x_test.shape[0], 'test samples.')
        print('Predicting {} days'.format(x_recent.shape[0]))
        print('Train sample shape: ', x_train.shape)
        print('Test sample shape: ', x_test.shape)
        print('Train label shape:', y_train.shape)
        print('Test label shape:', y_test.shape)
        # sample data statistics
        print('Sample Data: ')
        print(df.describe())

    return print_summary(x_train, x_test, x_recent, y_train, y_test, df)


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
    return 'do some magic!'


def dataset_graph_correlation_correlation_type_post(correlationType):  # noqa: E501
    """Returns a graph utilizing the selected correlation type

    1. Spearman, 2. Pearson, 3. Kendall # noqa: E501

    :param correlationType: Must be a number equal to 1, 2, or 3
    :type correlationType: int

    :rtype: InlineImage
    """
    #print('Choose Graph Type: [1]Trend Curve [2]Candlestick [3]Correlation Map')
    global choice
    choice = correlationType
    if choice == 1:
        visuals.plot_trend(currencies, coin_type, coin_feat)
    elif choice == 2:
        visuals.plot_candlestick(currencies, coin_type, coin_feat)
    elif choice == 3:
        print('Choose correlation type: [1]Spearman [2]Pearson [3]Kendall')
        corr_choice = int(input())
        if corr_choice == 1:
            return visuals.plot_correlation('spearman')
        elif corr_choice == 2:
            return visuals.plot_correlation('pearson')
        elif corr_choice == 3:
            return visuals.plot_correlation('kendall')
    return 'Error'


def dataset_graph_selection_graph_selection_post(graphSelection):  # noqa: E501
    """Selects a graph type

    1. Trend Curve, 2. Candlestick, 3. Correlation Map # noqa: E501

    :param graphSelection: Must be a number equal to 1, 2, or 3
    :type graphSelection: int

    :rtype: ApiResponse
    """
    global choice
    choice = graphSelection
    if choice == 1:
        visuals.plot_trend(currencies, coin_type, coin_feat)
    elif choice == 2:
        visuals.plot_candlestick(currencies, coin_type, coin_feat)
    elif choice == 3:
        print('Choose correlation type: [1]Spearman [2]Pearson [3]Kendall')
        corr_choice = int(input())
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
    choice = modelType

    if choice == 1:
        model = models.LR(x_train, y_train, x_test, y_test, x_recent)
        print('Linear Regression model selected.\n')
    elif choice == 2:
        model = models.SVR(x_train, y_train, x_test, y_test, x_recent)
        print('Support Vector Regression model selected.\n')
    elif choice == 3:
        model = models.MLP(x_train, y_train, x_test, y_test, x_recent)
        model.build()
        print('Multilayer Perceptron model selected.\n')
    elif choice == 4:
        model = models.GTBR(x_train, y_train, x_test, y_test, x_recent)
        print('Gradient Boosting Regression model selected.\n')
    return 'do some magic!'


def dataset_operation_type_operation_type_post(operationType):  # noqa: E501
    """Selects what operation to do, either training, testing, or prediction

    1. Train, 2. Test, 3. Predict # noqa: E501

    :param operationType: Must be a number equal to 1, 2, or 3
    :type operationType: int

    :rtype: ApiResponse
    """
    op = operationType

    if op == 1:
        print('Training initiated...\n')
        model.train()
    elif op == 2:
        print('Evaluating model on test data...\n')
        model.test()
    elif op == 3:
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
    print('Enter Currency File Name: ')
    # name = input()
    name = 'bitcoin_price.csv'

    x_train, x_test, x_recent, y_train, y_test, df = utils.load_data(input_dir, name)
    print('---------------------------------------')
    print(x_train.shape[0], 'training samples.')
    print(x_test.shape[0], 'test samples.')
    print('Predicting {} days'.format(x_recent.shape[0]))
    print('Train sample shape: ', x_train.shape)
    print('Test sample shape: ', x_test.shape)
    print('Train label shape:', y_train.shape)
    print('Test label shape:', y_test.shape)
    return 'do some magic!'
