# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.inline_image import InlineImage  # noqa: E501
from swagger_server.models.inline_response200 import InlineResponse200  # noqa: E501
from swagger_server.models.inline_response2001 import InlineResponse2001  # noqa: E501
from swagger_server.models.table import Table  # noqa: E501
from swagger_server.test import BaseTestCase


class TestDatasetController(BaseTestCase):
    """DatasetController integration test stubs"""

    def test_dataset_get_candlestick_graph_currency_name_post(self):
        """Test case for dataset_get_candlestick_graph_currency_name_post

        Returns a candlestick graph of the specified currency
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.0/dataset/getCandlestickGraph/{currencyName}'.format(currencyName='currencyName_example'),
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_get_cryptocurrency_correlation_graph_get(self):
        """Test case for dataset_get_cryptocurrency_correlation_graph_get

        Send specification about what data to graph
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.0/dataset/getCryptocurrencyCorrelationGraph',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_get_graph_close_all_time_get(self):
        """Test case for dataset_get_graph_close_all_time_get

        Returns a graph of all currencies closing values for all days we have data of
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.0/dataset/getGraphCloseAllTime',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_get_graph_close_one_year_get(self):
        """Test case for dataset_get_graph_close_one_year_get

        Returns a graph of all currencies closing values for one year
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.0/dataset/getGraphCloseOneYear',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_get_graph_graphdata_post(self):
        """Test case for dataset_get_graph_graphdata_post

        Unsupported Method
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.0/dataset/getGraph/{graphdata}'.format(graphdata='graphdata_example'),
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_get_graph_open_close_currency_name_post(self):
        """Test case for dataset_get_graph_open_close_currency_name_post

        Produces a graph with a specific currency's opening and closing values
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.0/dataset/getGraphOpenClose/{currencyName}'.format(currencyName='currencyName_example'),
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_get_graph_open_close_high_low_currency_name_post(self):
        """Test case for dataset_get_graph_open_close_high_low_currency_name_post

        Produces a graph with a specific currency's opening, closing, high, and low values.
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.0/dataset/getGraphOpenCloseHighLow/{currencyName}'.format(currencyName='currencyName_example'),
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_get_stats_get(self):
        """Test case for dataset_get_stats_get

        Get Statistics for the Dataset
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.0/dataset/getStats',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_get_table_table_name_post(self):
        """Test case for dataset_get_table_table_name_post

        Returns the specified table
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.0/dataset/getTable/{tableName}'.format(tableName='tableName_example'),
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
