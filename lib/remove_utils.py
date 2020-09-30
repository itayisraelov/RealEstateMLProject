import pandas as pd
from lib.constants import *


def data_cleansing(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function remove
    :param df: The data frame to remove the features from
    :type df:pandas.DataFrame
    :return: A new DataFrame without the
    :rtype:pandas.DataFrame
    """
    result = remove_sale_price_0(df)
    result = remove_redundant_features(result)
    remove_nan_outliers(result)
    result = remove_sale_price_outliers(result)
    result = remove_res_units_outliers(result)
    return result


def remove_outliers_zscore(df: pd.DataFrame, feature_name:str, z_threshold: float=1.96):
    """
    Remove the rows below or above the given z_score
    :param df:
    :type df:
    :param feature_name:
    :type feature_name:
    :param z_threshold:
    :type z_threshold:
    :return:
    :rtype:
    """
    mean = df[feature_name].mean()
    std = df[feature_name].std()
    dist = (df[feature_name] - mean) / std
    return df.loc[(dist > -z_threshold) & (dist < z_threshold)]


def remove_sale_price_0(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function remove the rows where sale price is 0
    :param df: The data frame to remove the rows from
    :type df:pandas.DataFrame
    :return: A new DataFrame without the relevant rows
    :rtype:pandas.DataFrame
    """
    return remove_outliers_zscore(df, "SALE PRICE")


def remove_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function remove the redundant features in our data
    :param df: The data frame to remove the features from
    :type df:pandas.DataFrame
    :return: A new DataFrame without the redundant values
    :rtype:pandas.DataFrame
    """
    return df.drop(REDUNDANT_FEATURES, axis=1)


def remove_nan_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function remove the nan rows in the data frame, where a value of nan
    is present & most of the data frame includes this value
    :param df: The data frame to remove the rows from
    :type df:pandas.DataFrame
    :return: A new DataFrame without the relevant rows
    :rtype:pandas.DataFrame
    """
    return df.dropna(how='any', axis=0)


def remove_sale_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :type df:pandas.DataFrame
    :return:
    :rtype:pandas.DataFrame
    """
    res = df[(df['SALE PRICE'] > int(1e5)) & (df['SALE PRICE'] < int(2e6))]
    return res


def remove_res_units_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :type df:pandas.DataFrame
    :return:
    :rtype:pandas.DataFrame
    """
    res = df[(df['RESIDENTIAL UNITS'] < 10.0)]
    return res





