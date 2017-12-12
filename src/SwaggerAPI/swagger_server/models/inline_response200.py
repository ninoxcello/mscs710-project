# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server import util


class InlineResponse200(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, count: float=None, mean: float=None, std: float=None, min: float=None, _25: float=None, _50: float=None, _75: float=None, max: float=None):  # noqa: E501
        """InlineResponse200 - a model defined in Swagger

        :param count: The count of this InlineResponse200.  # noqa: E501
        :type count: float
        :param mean: The mean of this InlineResponse200.  # noqa: E501
        :type mean: float
        :param std: The std of this InlineResponse200.  # noqa: E501
        :type std: float
        :param min: The min of this InlineResponse200.  # noqa: E501
        :type min: float
        :param _25: The _25 of this InlineResponse200.  # noqa: E501
        :type _25: float
        :param _50: The _50 of this InlineResponse200.  # noqa: E501
        :type _50: float
        :param _75: The _75 of this InlineResponse200.  # noqa: E501
        :type _75: float
        :param max: The max of this InlineResponse200.  # noqa: E501
        :type max: float
        """
        self.swagger_types = {
            'count': float,
            'mean': float,
            'std': float,
            'min': float,
            '_25': float,
            '_50': float,
            '_75': float,
            'max': float
        }

        self.attribute_map = {
            'count': 'count',
            'mean': 'mean',
            'std': 'std',
            'min': 'min',
            '_25': '25%',
            '_50': '50%',
            '_75': '75%',
            'max': 'max'
        }

        self._count = count
        self._mean = mean
        self._std = std
        self._min = min
        self.__25 = _25
        self.__50 = _50
        self.__75 = _75
        self._max = max

    @classmethod
    def from_dict(cls, dikt) -> 'InlineResponse200':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The inline_response_200 of this InlineResponse200.  # noqa: E501
        :rtype: InlineResponse200
        """
        return util.deserialize_model(dikt, cls)

    @property
    def count(self) -> float:
        """Gets the count of this InlineResponse200.


        :return: The count of this InlineResponse200.
        :rtype: float
        """
        return self._count

    @count.setter
    def count(self, count: float):
        """Sets the count of this InlineResponse200.


        :param count: The count of this InlineResponse200.
        :type count: float
        """

        self._count = count

    @property
    def mean(self) -> float:
        """Gets the mean of this InlineResponse200.


        :return: The mean of this InlineResponse200.
        :rtype: float
        """
        return self._mean

    @mean.setter
    def mean(self, mean: float):
        """Sets the mean of this InlineResponse200.


        :param mean: The mean of this InlineResponse200.
        :type mean: float
        """

        self._mean = mean

    @property
    def std(self) -> float:
        """Gets the std of this InlineResponse200.


        :return: The std of this InlineResponse200.
        :rtype: float
        """
        return self._std

    @std.setter
    def std(self, std: float):
        """Sets the std of this InlineResponse200.


        :param std: The std of this InlineResponse200.
        :type std: float
        """

        self._std = std

    @property
    def min(self) -> float:
        """Gets the min of this InlineResponse200.


        :return: The min of this InlineResponse200.
        :rtype: float
        """
        return self._min

    @min.setter
    def min(self, min: float):
        """Sets the min of this InlineResponse200.


        :param min: The min of this InlineResponse200.
        :type min: float
        """

        self._min = min

    @property
    def _25(self) -> float:
        """Gets the _25 of this InlineResponse200.


        :return: The _25 of this InlineResponse200.
        :rtype: float
        """
        return self.__25

    @_25.setter
    def _25(self, _25: float):
        """Sets the _25 of this InlineResponse200.


        :param _25: The _25 of this InlineResponse200.
        :type _25: float
        """

        self.__25 = _25

    @property
    def _50(self) -> float:
        """Gets the _50 of this InlineResponse200.


        :return: The _50 of this InlineResponse200.
        :rtype: float
        """
        return self.__50

    @_50.setter
    def _50(self, _50: float):
        """Sets the _50 of this InlineResponse200.


        :param _50: The _50 of this InlineResponse200.
        :type _50: float
        """

        self.__50 = _50

    @property
    def _75(self) -> float:
        """Gets the _75 of this InlineResponse200.


        :return: The _75 of this InlineResponse200.
        :rtype: float
        """
        return self.__75

    @_75.setter
    def _75(self, _75: float):
        """Sets the _75 of this InlineResponse200.


        :param _75: The _75 of this InlineResponse200.
        :type _75: float
        """

        self.__75 = _75

    @property
    def max(self) -> float:
        """Gets the max of this InlineResponse200.


        :return: The max of this InlineResponse200.
        :rtype: float
        """
        return self._max

    @max.setter
    def max(self, max: float):
        """Sets the max of this InlineResponse200.


        :param max: The max of this InlineResponse200.
        :type max: float
        """

        self._max = max