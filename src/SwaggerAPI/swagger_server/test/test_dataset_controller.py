# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.api_response import ApiResponse  # noqa: E501
from swagger_server.models.inline_image import InlineImage  # noqa: E501
from swagger_server.models.inline_response200 import InlineResponse200  # noqa: E501
from swagger_server.models.table import Table  # noqa: E501
from swagger_server.test import BaseTestCase


class TestDatasetController(BaseTestCase):
    """DatasetController integration test stubs"""

    def test_dataset_currency_selection_currency_name_post(self):
        """Test case for dataset_currency_selection_currency_name_post

        Selects a specific currency type
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.2/dataset/currencySelection/{currencyName}'.format(currencyName='currencyName_example'),
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_get_stats_get(self):
        """Test case for dataset_get_stats_get

        Get Statistics for the Dataset
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.2/dataset/getStats',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_get_table_table_name_post(self):
        """Test case for dataset_get_table_table_name_post

        Returns the specified table
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.2/dataset/getTable/{tableName}'.format(tableName='tableName_example'),
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_graph_correlation_correlation_type_post(self):
        """Test case for dataset_graph_correlation_correlation_type_post

        Returns a graph utilizing the selected correlation type
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.2/dataset/graphCorrelation/{correlationType}'.format(correlationType=56),
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_graph_selection_graph_selection_post(self):
        """Test case for dataset_graph_selection_graph_selection_post

        Selects a graph type
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.2/dataset/graphSelection/{graphSelection}'.format(graphSelection=56),
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_model_type_model_type_post(self):
        """Test case for dataset_model_type_model_type_post

        Selects a model type
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.2/dataset/modelType/{modelType}'.format(modelType=56),
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_operation_type_operation_type_post(self):
        """Test case for dataset_operation_type_operation_type_post

        Selects what operation to do, either training, testing, or prediction
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.2/dataset/operationType/{operationType}'.format(operationType=56),
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_dataset_train_currency_name_post(self):
        """Test case for dataset_train_currency_name_post

        Trains based on the currency selected
        """
        response = self.client.open(
            '/mmaffa/MoneyREST/1.0.2/dataset/train/{currencyName}'.format(currencyName='currencyName_example'),
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
