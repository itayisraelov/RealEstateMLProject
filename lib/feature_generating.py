import pandas as ps


def add_final_address(df: ps.DataFrame) -> None:
    """
    This function adds IN_PLACE!!! the FINAL_ADDRESS feature to the dataframe
    :param df: The data frame to add the features to
    :type df:pandas.DataFrame
    :return: A new DataFrame with the new feature
    :rtype:pandas.DataFrame
    """
    df['FINAL_ADDRESS'] = df['ADDRESS'].apply(lambda x: x + " NYC")


