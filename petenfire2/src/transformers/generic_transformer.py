"""Generic class for transformations"""
from typing import Callable, List, Union

import category_encoders as ce
import pandas as pd
from pandas.api.types import is_integer_dtype
from sklearn.base import BaseEstimator, TransformerMixin

import utils
from transformers.transformation_values_enum import TransformationValuesEnum
from transformers.transformations_enum import TransformationsEnum


class GenericTransformer(BaseEstimator, TransformerMixin):
    """Class used for data transformations"""

    def __init__(self, config: dict):
        self._config = config
        self._values = {}
        self._features_map = {}
        self._logger = utils.get_logger()

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        """Set the information in the instance needed for transformations

        Args:
            x (pd.DataFrame): the dataframe to fit with
            y (pd.Series, optional): the objective variable and its values. Defaults to None.

        Raises:
            KeyError: Incorrect configuration

        Returns:
            GenericTransformer: the transformation needed
        """
        if not self._config:
            return self

        if x is None:
            return self

        x_copy = x.copy()
        for key in self._config.keys():
            if key not in x:
                raise KeyError(f'{key} not in dataset')

            self._logger.debug('Fit column %s', key)
            self._fit_column(x_copy, y, key)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """With information saved, transform data

        Args:
            x (pd.DataFrame): the data to transform
            y (pd.Series, optional): the objective variable. Defaults to None.

        Raises:
            KeyError: not information found on instance

        Returns:
            pd.DataFrame: the transformed data
        """
        if not self._config:
            return x

        if x is None:
            return x

        x_copy = x.copy()
        for key in self._config.keys():
            if key not in x_copy:
                raise KeyError(f'{key} not in dataset')

            self._logger.debug('Transform column %s', key)
            x_copy = self._transform_column(x_copy, key)

        return x_copy

    def _fit_column(
            self, data: pd.DataFrame, objective: pd.Series, column: str):
        """Save necessary for a column transformation

        Args:
            data (pd.DataFrame): the data to use as base
            objective (pd.Series): the values of the objective column
            column (str): the column that is being fitted with

        Raises:
            KeyError: invalid configuration
        """
        transformation_config = self._config[column]
        fitted_transformations = []
        column_data = data[column]
        for transformation in transformation_config.keys():
            if len(fitted_transformations) > 0:
                column_data = self._apply_previous_transformations(
                    data, column, fitted_transformations
                )
                data[column] = column_data

            fitted_transformations.append(transformation)
            if transformation == TransformationsEnum.REPLACE_NAN:
                self._save_feature_map(column, [column])
            elif transformation == TransformationsEnum.TRUNCATE_STD:
                self._fit_truncate_std(column_data, column)
            elif transformation == TransformationsEnum.IMPUTE:
                self._fit_impute(
                    column_data, column, transformation_config[transformation]
                )
            elif transformation == TransformationsEnum.MIN_MAX:
                self._save_feature_map(column, [column])
            elif transformation == TransformationsEnum.MANUAL_MAPPING:
                self._fit_ordinal_with_mapping(
                    column_data, transformation_config[transformation], column)
            elif transformation == TransformationsEnum.OTHER_GROUP:
                self._save_feature_map(column, [column])
            elif transformation == TransformationsEnum.TO_BINARY:
                self._save_to_binary_feature_map(
                    column, transformation_config[transformation]
                )
            elif transformation == TransformationsEnum.ENCODER:
                self._fit_encoder(
                    column_data, objective, column,
                    transformation_config[transformation]
                )
            else:
                raise KeyError(f'Not a valid transformation: {transformation}')

    def _save_to_binary_feature_map(self, column: str, config: dict):
        """Save on _features_map the value for the binary feature map

        Args:
            column (str): the name of the column
            config (dict): the configuration of to_binary
        """
        self._save_feature_map(column, [config['column_name']])

    def _save_feature_map(self, column: str, new_names: List[str]):
        """Save a column features in _features_map

        Args:
            column (str): the name of the column
            new_names (List[str]): the list of columns it maps to
        """
        self._features_map[column] = new_names

    def _apply_previous_transformations(
            self, data: pd.DataFrame, column: str,
            transformations: List[TransformationsEnum]) -> pd.Series:
        """To fit sometimes you have to apply transformations before, this
        applies the previous transformations needed

        Args:
            data (pd.DataFrame): the dataset used to transform
            column (str): the name of the column
            transformations (List[TransformationsEnum]): the list of transformations to apply

        Returns:
            pd.Series: the transformed values
        """
        if not transformations:
            return data

        df_transformed = self._transform_column_from_config(
            data, column, transformations
        )
        return df_transformed[column]

    def _get_values(self, column: str) -> dict:
        """From the values saved, get the values of a column

        Args:
            column (str): the column to get the vaues from

        Returns:
            dict: the values of a column
        """
        return self._values.get(column, {})

    def _fit_truncate_std(self, data: pd.Series, column: str):
        """Save the values needed to truncate using the standar deviation

        Args:
            data (pd.Series): the data
            column (str): the column to set the values to
        """
        current_values = self._get_values(column)
        self._get_value_if_not_set(
            current_values, TransformationValuesEnum.STANDAR_DEVIATION, self._get_std,
            data, column
        )

        self._get_value_if_not_set(
            current_values, TransformationValuesEnum.QUANTILE_1, self._get_quantile1,
            data, column
        )

        self._get_value_if_not_set(
            current_values, TransformationValuesEnum.QUANTILE_3, self._get_quantile3,
            data, column
        )

        self._values[column] = current_values

    def _set_value_if_not_set(
            self, current_values: dict, value_key: TransformationValuesEnum,
            func: Callable, data: pd.Series, column: str):
        """Set the value in the instance values if it hasn't been calculated yet

        Args:
            current_values (dict): the values to check and update
            value_key (TransformationValuesEnum): the key to update
            func (Callable): the function to get the valeu from
            data (pd.Series): the data used to get the value
            column (str): the name of the data
        """
        self._get_value_if_not_set(
            current_values, value_key, func, data, column)

        self._values[column] = current_values

    def _get_value_if_not_set(
            self, current_values: dict, value_key: TransformationValuesEnum,
            func: Callable, data: pd.Series, column: str, **kwargs):
        """Set the value in the instance values if it hasn't been calculated yet

        Args:
            current_values (dict): the values to check and update
            value_key (TransformationValuesEnum): the key to update
            func (Callable): the function to get the valeu from
            data (pd.Series): the data used to get the value
            column (str): the name of the column
        """
        if value_key in current_values:
            return

        no_column_values = [
            TransformationValuesEnum.STANDAR_DEVIATION,
            TransformationValuesEnum.QUANTILE_1,
            TransformationValuesEnum.QUANTILE_3,
            TransformationValuesEnum.MEAN,
            TransformationValuesEnum.MODE,
            TransformationValuesEnum.MEDIAN
        ]
        if value_key in no_column_values:
            current_values[value_key] = func(data, **kwargs)
        else:
            current_values[value_key] = func(data, column, **kwargs)

    def _get_std(self, data: pd.Series) -> float:
        """Get the standar deviation of the data

        Args:
            data (pd.Series): data used to get the value

        Returns:
            float: the standar deviation
        """
        return data.std()

    def _get_quantile1(self, data: pd.Series) -> float:
        """Get the quantile 1 (0.25)

        Args:
            data (pd.Series): data used to get the value

        Returns:
            float: the quantile 1
        """
        return data.quantile(0.25)

    def _get_quantile3(self, data: pd.Series) -> float:
        """Get the quantile 3 (0.75)

        Args:
            data (pd.Series): data used to get the value

        Returns:
            float: the quantile 3
        """
        return data.quantile(0.75)

    def _get_mode(self, data: pd.Series) -> float:
        """Get the mode

        Args:
            data (pd.Series): data used to get the value
            column (str): the name of

        Returns:
            float: the mode
        """
        return data.mode().values[0]

    def _keep_dtype(self, data: pd.Series, value: Union[float, int]) -> float:
        """Check the original data type and try to keep that data type

        Args:
            data (pd.Series): the values of the column
            value (Union[float, int]): the value to impute

        Returns:
            float: the value to save on _values
        """
        if is_integer_dtype(data):
            return int(value)

        return value

    def _get_median(self, data: pd.Series) -> float:
        """Get the median of the values

        Args:
            data (pd.Series): data used to get the value

        Returns:
            float: the median
        """
        return self._keep_dtype(data, data.median())

    def _get_mean(self, data: pd.Series) -> float:
        """Get the mean

        Args:
            data (pd.Series): data used to get the value

        Returns:
            float: the mean of the data
        """
        return self._keep_dtype(data, data.mean())

    def _fit_impute(self, data: pd.Series, column: str, impute_with: str):
        """Set the necessary information to impute values

        Args:
            data (pd.Series): the data to use as base
            column (str): the column that is being processed
            impute_with (str): the value to be used to impute
        """
        current_values = self._get_values(column)

        self._save_feature_map(column, [column])

        if impute_with == TransformationValuesEnum.MODE:
            self._set_value_if_not_set(
                current_values, TransformationValuesEnum.MODE, self._get_mode,
                data, column
            )
            return

        if impute_with == TransformationValuesEnum.MEAN:
            self._set_value_if_not_set(
                current_values, TransformationValuesEnum.MEAN, self._get_mean,
                data, column
            )
            return

        if impute_with == TransformationValuesEnum.MEDIAN:
            self._set_value_if_not_set(
                current_values, TransformationValuesEnum.MEDIAN, self._get_median,
                data, column
            )
            return

    def _fit_ordinal_with_mapping(
            self, data: pd.DataFrame, mapping: dict, column: str):
        """Fit an ordinal encoder with an defined mapping

        Args:
            data (pd.DataFrame): the data used to fit
            mapping (dict): tha stablished values to map with
            column (str): the name of the column to fit
        """
        current_values = self._get_values(column)

        self._get_value_if_not_set(
            current_values,
            TransformationValuesEnum.ORDINAL,
            self._get_fitted_encoder, data,
            column,
            **{
                'encoder': TransformationValuesEnum.ORDINAL,
                'mapping': [
                    {
                        'col': column,
                        'mapping': mapping
                    }
                ]
            }
        )

        self._values[column] = current_values

    def _get_fitted_encoder(
            self, data: pd.DataFrame, column: str,
            encoder: TransformationValuesEnum,
            **kwargs) -> Union[ce.OrdinalEncoder, ce.OneHotEncoder,
                               ce.TargetEncoder]:
        """Create an encoder, fit, and return it

        Args:
            data (pd.DataFrame): the data used to fit
            column (str): the name of the column
            encoder (TransformationValuesEnum): the name of the encoder

        Raises:
            KeyError: the column to fit doesn't exist

        Returns:
            Union[ce.OrdinalEncoder, ce.OneHotEncoder,
            ce.TargetEncoder]: the fitted encoder
        """
        if encoder == TransformationValuesEnum.ORDINAL:
            encoder = ce.OrdinalEncoder(
                handle_unknown='value', handle_missing='value',
                **kwargs
            )
            encoder.fit(data)

            self._save_feature_map(column, [column])

            return encoder

        if encoder == TransformationValuesEnum.ONE_HOT:
            return self._get_one_hot_encoder(data, column)

        if encoder == TransformationValuesEnum.TARGET:
            return self._get_target_encoder(data, column, kwargs['objective'])

        raise KeyError(f'{encoder} not implemented')

    def _get_one_hot_encoder(
            self, data: pd.DataFrame, column: str) -> ce.OneHotEncoder:
        """Create and fit a one hot encoder

        Args:
            data (pd.DataFrame): the data used to fit
            column (str): the name of the column to fit

        Returns:
            ce.OneHotEncoder: the fitted encoder
        """
        encoder = ce.OneHotEncoder(cols=[column], use_cat_names=True)
        encoder.fit(data)

        self._save_feature_map(
            column, encoder.get_feature_names_out().tolist()
        )

        return encoder

    def _get_target_encoder(
            self, data: pd.Series, column: str, objective: pd.Series) -> ce.TargetEncoder:
        """Create and fit a target encoder

        Args:
            data (pd.Series): the data used to fit
            column (str): the name of the column used to create the data
            objective (pd.Series): the values of the objective variables

        Returns:
            ce.TargetEncoder: the fitted encoder
        """
        encoder = ce.TargetEncoder(cols=[column])
        encoder.fit(data, objective)

        self._save_feature_map(column, [column])

        return encoder

    def _fit_encoder(
            self, data: pd.DataFrame, objective: pd.Series, column: str,
            encoder: TransformationValuesEnum):
        """Create, fit encoder, and set it as a value of the tranformer

        Args:
            data (pd.DataFrame): the data to fit with
            objective (pd.Series): the values of the objective variable
            column (str): the name of the column to encode
            encoder (TransformationValuesEnum): the type of encoder to use
        """
        current_values = self._get_values(column)

        self._logger.debug('Fit encoder %s for column %s', encoder, column)
        self._get_value_if_not_set(
            current_values,
            encoder,
            self._get_fitted_encoder, data,
            column,
            **{
                'encoder': encoder,
                'objective': objective,
            }
        )

        self._values[column] = current_values

    def _transform_column(
            self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply transformation to a column

        Args:
            data (pd.DataFrame): the data of a column
            column (str): the name of the column

        Raises:
            KeyError: invalid transformation

        Returns:
            pd.DataFrame: the transformed data
        """
        transformation_config = self._config[column]
        return self._transform_column_from_config(
            data, column, transformation_config.keys()
        )

    def _transform_column_from_config(
            self, data: pd.DataFrame, column: str,
            transformations: List[TransformationsEnum]) -> pd.DataFrame:
        """Apply a transformation to a column

        Args:
            data (pd.DataFrame): the data used to train
            column (str): the name of the column
            transformations (List[TransformationsEnum]): the list of transformations to apply

        Raises:
            KeyError: if the key doesn't exist

        Returns:
            pd.DataFrame: the transformed data
        """
        transformation_config = self._config[column]
        for transformation in transformations:
            if transformation == TransformationsEnum.REPLACE_NAN:
                data = self._transform_replace_nan(
                    data, column, transformation_config[transformation]
                )

            elif transformation == TransformationsEnum.TRUNCATE_STD:
                data = self._transform_truncate_std(data, column)

            elif transformation == TransformationsEnum.IMPUTE:
                data = self._transform_impute(
                    data, column, transformation_config[transformation]
                )

            elif transformation == TransformationsEnum.MIN_MAX:
                data = self._transform_min_max(data, column)

            elif transformation == TransformationsEnum.MANUAL_MAPPING:
                data = self._transform_encoder(
                    data, TransformationValuesEnum.ORDINAL, column
                )

            elif transformation == TransformationsEnum.OTHER_GROUP:
                data = self._transform_other_group(
                    data, column, transformation_config[transformation]['keep'],
                    transformation_config[transformation]['default']
                )

            elif transformation == TransformationsEnum.TO_BINARY:
                data = self._transform_to_binary(
                    data, column,
                    transformation_config[transformation]['values_as_true'],
                    transformation_config[transformation]['column_name']
                )
            elif transformation == TransformationsEnum.ENCODER:
                data = self._transform_encoder(
                    data, transformation_config[transformation], column
                )

            else:
                raise KeyError(f'Not a valid transformation: {transformation}')

        return data

    def _transform_encoder(
            self, data: pd.DataFrame, encoder: str, column: str) -> pd.DataFrame:
        """Do a transformation for an encoder

        Args:
            data (pd.DataFrame): all the data
            encoder (str): the encoder used to find in values
            column (str): the column of the data

        Returns:
            pd.DataFrame: the transformed data
        """
        self._logger.debug('Encode with %s column %s', encoder, column)
        transformer = self._values[column][encoder]

        df_transformed = transformer.transform(data[column])
        data = data.drop(columns=[column])
        data = pd.concat([data, df_transformed], axis='columns')

        return data

    def _transform_other_group(
            self, data: pd.DataFrame, column: str, possible_values: List[str],
            default_value: str) -> pd.DataFrame:
        """Apply the other group transformation

        Args:
            data (pd.DataFrame): all the data
            column (str): the name of the column
            possible_values (List[str]): the list of values that will keep
            default_value (str): the value for the other values

        Returns:
            pd.DataFrame: the transformed data
        """
        data[column] = data[column].mask(
            ~(data[column].isin(possible_values)),
            default_value)

        return data

    def _transform_to_binary(
            self, data: pd.DataFrame, column: str, possible_values: List[str],
            new_column_name: str) -> pd.DataFrame:
        """Transform a categorical column to a binary column

        Args:
            data (pd.DataFrame): all the data
            column (str): the name of the column to transform
            possible_values (List[str]): the possible values that will be true
            new_column_name (str): the name of the new column

        Returns:
            pd.DataFrame: the transformed data
        """
        data[column] = data[column].isin(possible_values).astype(int)
        data = data.rename(columns={column: new_column_name})
        return data

    def _transform_replace_nan(
            self, data: pd.DataFrame, column: str,
            value: Union[float, int, str]) -> pd.DataFrame:
        """Replace nana values with a fixed value

        Args:
            data (pd.DataFrame): the data to transform
            column (str): the name of the column to transform
            value (Union[float, int, str]): the value used to replace nan

        Returns:
            pd.DataFrame: the transformed data
        """
        data[column] = data[column].fillna(value)
        return data

    def _transform_truncate_std(
            self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Truncate values based on the standar deviation

        Args:
            data (pd.DataFrame): the data to transform
            column (str): the column to get the values from

        Returns:
            pd.DataFrame: the transformed data
        """
        std_amount = self._config[column][
            TransformationsEnum.TRUNCATE_STD] * self._values[column][
            TransformationValuesEnum.STANDAR_DEVIATION]
        lower_bound = self._values[column][
            TransformationValuesEnum.QUANTILE_1] - std_amount
        upper_bound = self._values[column][
            TransformationValuesEnum.QUANTILE_3] + std_amount

        # Keep data type
        if is_integer_dtype(data[column]):
            lower_bound = int(lower_bound)
            upper_bound = int(upper_bound)

        df_column = data[column]
        data[column] = self._limit_values(
            df_column, lower_bound, upper_bound, lower_bound, upper_bound
        )
        return data

    def _limit_values(
            self, data: pd.DataFrame, lower_bound: Union[int, float],
            upper_bound: Union[int, float], min_replace: Union[int, float],
            max_replace: Union[int, float]) -> pd.DataFrame:
        """Limit values based on a min and max values

        Args:
            data (pd.DataFrame): the data to transform
            lower_bound (Union[int, float]): the min value the data will have
            upper_bound (Union[int, float]): the max value the data will have
            min_replace (Union[int, float]): the value used when the data is
            lower than the upper bound
            max_replace (Union[int, float]): the value used when the data is
            greater than the upper bound

        Returns:
            pd.DataFrame: the transformed data
        """
        truncated = data.mask(data < lower_bound, min_replace)
        return truncated.mask(data > upper_bound, max_replace)

    def _transform_impute(
            self, data: pd.DataFrame, column: str,
            impute_with: dict) -> pd.DataFrame:
        """Fill nan values with a stored value

        Args:
            data (pd.DataFrame): the data to transform
            column (str): the column used to get the value to impute with
            impute_with (dict): the type of imputation

        Returns:
            pd.DataFrame: the transformed data
        """
        data[column] = data[column].fillna(self._values[column][impute_with])
        return data

    def _transform_min_max(
            self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Change values to have a min and max value

        Args:
            data (pd.DataFrame): the data to transform
            column (str): the column used to get min and max values

        Returns:
            pd.DataFrame: the transformed data
        """

        min_max_config = self._config[column][TransformationsEnum.MIN_MAX]
        lower_bound = min_max_config[TransformationValuesEnum.MINIMUN]
        upper_bound = min_max_config[TransformationValuesEnum.MAXIMUM]

        max_value = upper_bound
        min_value = lower_bound
        if TransformationsEnum.MIN_MAX_REPLACE in min_max_config:
            if TransformationValuesEnum.MAXIMUM in min_max_config[
                    TransformationsEnum.MIN_MAX_REPLACE]:
                max_value = min_max_config[
                    TransformationsEnum.MIN_MAX_REPLACE][
                    TransformationValuesEnum.MAXIMUM]
            elif TransformationValuesEnum.MINIMUN in min_max_config[
                    TransformationsEnum.MIN_MAX_REPLACE]:
                min_value = min_max_config[
                    TransformationsEnum.MIN_MAX_REPLACE][
                    TransformationValuesEnum.MINIMUN]

        data[column] = self._limit_values(
            data[column], lower_bound, upper_bound, min_value, max_value
        )
        return data

    def get_features_map(self) -> dict:
        """Return the features map saved after a fit

        Returns:
            dict: a dictionary with the key as the name of the column and the
            value as a list of the new feature names
        """
        return self._features_map
