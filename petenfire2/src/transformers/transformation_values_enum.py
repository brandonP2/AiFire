"""Possible values to save for the transformations"""
from enum import Enum


class TransformationValuesEnum(str, Enum):
    """Class for possible values to save on the generic transformer"""
    STANDAR_DEVIATION = 'std'
    QUANTILE_1 = 'q1'
    QUANTILE_3 = 'q3'
    MODE = 'mode'
    MEAN = 'mean'
    MEDIAN = 'median'
    MINIMUN = 'min'
    MAXIMUM = 'max'
    ORDINAL = 'ordinal'
    ONE_HOT = 'one_hot'
    TARGET = 'target'
