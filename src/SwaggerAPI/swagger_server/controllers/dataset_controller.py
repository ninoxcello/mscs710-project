import connexion
import six

from swagger_server.models.api_response import ApiResponse  # noqa: E501
from swagger_server.models.inline_image import InlineImage  # noqa: E501
from swagger_server.models.inline_response200 import InlineResponse200  # noqa: E501
from swagger_server.models.table import Table  # noqa: E501
from swagger_server import util


def dataset_currency_selection_currency_name_post(currencyName):  # noqa: E501
    """Selects a specific currency type

    The currency type should be either bitcoin, bitconnect, dash, ethereum, iota, litecoin, monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, or waves # noqa: E501

    :param currencyName: Should be equal to either bitcoin, bitconnect, dash, ethereum, iota, litecoin, monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, or waves
    :type currencyName: str

    :rtype: ApiResponse
    """
    return 'do some magic!'


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

    :param correlationType: information
    :type correlationType: str

    :rtype: InlineImage
    """
    return 'do some magic!'


def dataset_graph_selection_graph_selection_post(graphSelection):  # noqa: E501
    """Selects a graph type

    1. Trend Curve, 2. Candlestick, 3. Correlation Map # noqa: E501

    :param graphSelection: Must be a value equal to 1, 2, or 3
    :type graphSelection: str

    :rtype: ApiResponse
    """
    return 'do some magic!'


def dataset_model_type_model_type_post(modelType):  # noqa: E501
    """Selects a model type

    1. Linear Regression, 2. Support Vector Regression, 3. Multilayer Perceptron # noqa: E501

    :param modelType: Should be a number equal to 1, 2, or 3
    :type modelType: int

    :rtype: ApiResponse
    """
    return 'do some magic!'


def dataset_operation_type_operation_type_post(operationType):  # noqa: E501
    """Selects what operation to do, either training, testing, or prediction

    1. Train, 2. Test, 3. Predict # noqa: E501

    :param operationType: information
    :type operationType: str

    :rtype: ApiResponse
    """
    return 'do some magic!'


def dataset_train_currency_name_post(currencyName):  # noqa: E501
    """Trains based on the currency selected

    Should be either bitcoin, bitconnect, dash, ethereum, iota, litecoin, monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, or waves # noqa: E501

    :param currencyName: Should be equal to either bitcoin, bitconnect, dash, ethereum, iota, litecoin, monero, nem, neo, numeraire, omisego, qtum, ripple, stratis, or waves
    :type currencyName: str

    :rtype: ApiResponse
    """
    return 'do some magic!'
