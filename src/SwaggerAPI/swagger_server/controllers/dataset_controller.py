import connexion
import six

from swagger_server.models.inline_image import InlineImage  # noqa: E501
from swagger_server.models.inline_response200 import InlineResponse200  # noqa: E501
from swagger_server.models.inline_response2001 import InlineResponse2001  # noqa: E501
from swagger_server.models.table import Table  # noqa: E501
from swagger_server import util


def dataset_get_candlestick_graph_currency_name_post(currencyName):  # noqa: E501
    """Returns a candlestick graph of the specified currency

    Sends data about what the graph should show and returns with an image of the graph # noqa: E501

    :param currencyName: 
    :type currencyName: str

    :rtype: InlineImage
    """
    return 'do some magic!'


def dataset_get_cryptocurrency_correlation_graph_get():  # noqa: E501
    """Send specification about what data to graph

    Sends data about what the graph should show and returns with an image of the graph # noqa: E501


    :rtype: InlineImage
    """
    return 'do some magic!'


def dataset_get_graph_close_all_time_get():  # noqa: E501
    """Returns a graph of all currencies closing values for all days we have data of

    Sends data about what the graph should show and returns with an image of the graph # noqa: E501


    :rtype: InlineImage
    """
    return 'do some magic!'


def dataset_get_graph_close_one_year_get():  # noqa: E501
    """Returns a graph of all currencies closing values for one year

    Sends data about what the graph should show and returns with an image of the graph # noqa: E501


    :rtype: InlineImage
    """
    return 'do some magic!'


def dataset_get_graph_graphdata_post(graphdata):  # noqa: E501
    """Unsupported Method

    Sends data about what the graph should show and returns with an image of the graph # noqa: E501

    :param graphdata: 
    :type graphdata: str

    :rtype: InlineResponse200
    """
    return 'do some magic!'


def dataset_get_graph_open_close_currency_name_post(currencyName):  # noqa: E501
    """Produces a graph with a specific currency&#39;s opening and closing values

    Sends data about what the graph should show and returns with an image of the graph # noqa: E501

    :param currencyName: 
    :type currencyName: str

    :rtype: InlineImage
    """
    return 'do some magic!'


def dataset_get_graph_open_close_high_low_currency_name_post(currencyName):  # noqa: E501
    """Produces a graph with a specific currency&#39;s opening, closing, high, and low values.

    Sends data about what the graph should show and returns with an image of the graph # noqa: E501

    :param currencyName: 
    :type currencyName: str

    :rtype: InlineImage
    """
    return 'do some magic!'


def dataset_get_stats_get():  # noqa: E501
    """Get Statistics for the Dataset

    Returns values of count, mean, median, mode, min, max, standard deviation (std) for the dataset # noqa: E501


    :rtype: InlineResponse2001
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
