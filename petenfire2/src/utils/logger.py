"""General utilities"""
import logging
import os
import shutil
import sys
from logging import Logger
from typing import Union

import pandas as pd


def get_module_name() -> str:
    """Get the name of the module

    Returns:
        str: the name of the logger
    """
    return 'petenfire2'


def get_logger() -> Logger:
    """Get the logger for this project

    Returns:
        Logger: the logger of the project
    """
    return logging.getLogger(get_module_name())


def init_logger(level: Union[bool, str] = "info") -> Logger:
    """Create the logger for the script

    Args:
        level (Union[bool, str], optional): if true the log level will be debug, if
        false will be info; if string try to interpret as level name (info,
        debug, error). Defaults to "info".

    Returns:
        Logger: the logger to use to log details
    """
    logger = get_logger()

    logger.setLevel(logging.INFO)
    if isinstance(level, str):
        level = level.upper()

        logging_level = logging.getLevelName(level)

        if isinstance(logging_level, int):
            logger.setLevel(logging_level)

    elif level:
        logger.setLevel(logging.DEBUG)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    # Set formatter
    formatter = logging.Formatter(
        '%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s')
    handler.setFormatter(formatter)
    logger.handlers = [handler]

    return logger


def folder_files_to_df(
        folder_path: str, header: str = 'infer',
        dtypes: dict = None, file_type: str = 'CSV') -> pd.DataFrame:
    """From a folder, loop through the files and join the files in a DF

    Args:
        folder_path (str): the path to the directory
        header (str, optional): if header is explicit or implicit
        header (str, optional): the dtypes to load the dataframes

    Returns:
        pd.DataFrame: the dataframe with the information of all the files in
        the folder
    """
    logger = get_logger()
    logger.debug('Folder to look files in: %s', folder_path)
    if not folder_path:
        return None

    df = None
    files_list = sorted(os.listdir(folder_path))
    for index, filename in enumerate(files_list):
        f = os.path.join(folder_path, filename)

        df_current = None
        if os.path.isdir(f):
            folder_name = f.split('/')[-1]
            if folder_name.startswith('.'):
                continue

            # Look inside folder for files
            df_current = folder_files_to_df(f, header=header)

        elif os.path.isfile(f):
            # Read the file into DF
            logger.debug('Adding file: %s to df', f)
            if file_type == 'CSV':
                df_current = pd.read_csv(f, header=header)
            elif file_type == 'PARQUET':
                df_current = pd.read_parquet(f)

            if dtypes:
                df_current = df_current.astype(dtypes)

        if df_current is not None:
            # Add file to all data data frame
            if index == 0:
                df = df_current
            else:
                df = pd.concat([
                    df,
                    df_current])

    return df


def delete_files_in_folder(folder: str):
    """Delete the files in a folder

    Args:
        folder (str): the folder to delete the contents from
    """
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def transform_to_basic_type(df: pd.DataFrame) -> pd.DataFrame:
    """Transform object type number types to basic types

    Args:
        df (pd.DataFrame): the data frame to transform

    Returns:
        pd.DataFrame: the data frame with the basic types
    """
    map_types = {
        'Int32': 'int32',
        'Int64': 'int64',
        'Float32': 'float32',
        'Float64': 'float64',
    }
    df_basic_type = df.copy()
    for old_type, new_type in map_types.items():
        for col in df_basic_type.select_dtypes(include=old_type).columns:
            df_basic_type[col] = df_basic_type[col].astype(new_type)

    return df_basic_type
