"""Possible transformations to do"""
from enum import Enum


class TransformationsEnum(str, Enum):
    """Class for possible transformations configurations"""
    REPLACE_NAN = 'replace_nan'
    TRUNCATE_STD = 'truncate_std'
    IMPUTE = 'impute'
    MIN_MAX = 'min_max'
    MANUAL_MAPPING = 'manual_mapping'
    OTHER_GROUP = 'other_group'
    TO_BINARY = 'to_binary'
    ENCODER = 'encoder'
    MIN_MAX_REPLACE = 'replace'
