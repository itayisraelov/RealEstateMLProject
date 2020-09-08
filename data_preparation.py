import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from constants import EXCEL_EXTENSION
from sklearn.model_selection import train_test_split
from geopy.geocoders import Nominatim
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

color = sns.color_palette()
pd.options.mode.chained_assignment = None  # default='warn'


def main():
    df1, df2, df3, df4, df5 = load_data_from_excel()

    df = feature_engineering(df1, df2, df3, df4, df5)

    # get_some_information_from_data(df)

    df = remove_outliers(df)

    # get_some_information_from_data(df)

    df = complete_missing_values(df)

    train, train_target, test, test_target = split_train_test_data(df)

    final = convert_categories_to_numbers(train, test)

    final = normalization(final)

    train = final[:len(train)]
    test = final[len(train):]

    first_prediction_using_Random_Forest_Regressor(train, train_target, test, test_target)
    first_prediction_using_LGBM_Regressor(train, train_target, test, test_target)
    first_prediction_using_XGB_Regressor(train, train_target, test, test_target)


    # get_some_information_from_data(df)

    # plot_some_graph(df)
    print(f'Train data has {df.shape[0]} rows and {df.shape[1]} colummns')


def first_prediction_using_XGB_Regressor(train, train_target, test, test_target):
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                              max_depth=5, alpha=10, n_estimators=10)
    xg_reg.fit(train, train_target)
    res_pred = xg_reg.predict(test)
    rms = np.sqrt(mean_squared_error(test_target, res_pred))
    print("RMS: %f" % rms)


def first_prediction_using_LGBM_Regressor(train, train_target, test, test_target):
    lgbm = lgb.LGBMRegressor(max_depth=15, num_leaves=40)
    lgbm.fit(train, train_target)
    res_pred = lgbm.predict(test)
    rms = np.sqrt(mean_squared_error(test_target, res_pred))
    print("RMS: %f" % rms)


def first_prediction_using_Random_Forest_Regressor(train, train_target, test, test_target):
    rf = RandomForestRegressor(n_estimators=300, verbose=True, max_depth=10, n_jobs=-1)
    rf.fit(train, train_target)
    res_pred = rf.predict(test)
    rms = np.sqrt(mean_squared_error(test_target, res_pred))
    print("RMS: %f" % rms)


def split_train_test_data(df):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    train_target = df_train['SALE PRICE']
    train = df_train.drop(['SALE PRICE'], axis=1)
    test_target = df_test['SALE PRICE']
    test = df_test.drop(['SALE PRICE'], axis=1)
    return train, train_target, test, test_target


def normalization(final):
    for col in final.columns:
        minmax = MinMaxScaler()
        final[col] = minmax.fit_transform(final[col].values.reshape(-1, 1))
    return final


def convert_categories_to_numbers(df_train, df_test):
    cat = ['NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'BUILDING CLASS AS OF FINAL ROLL 18/19', 'ADDRESS',
           'BUILDING CLASS AT TIME OF SALE', 'SALE DATE', 'AREA', 'FINAL_ADDRESS']
    final = pd.concat([df_train, df_test])
    for col in cat:
        lb = LabelEncoder()
        final[col] = lb.fit_transform(final[col].values)

    final['TAX CLASS AS OF FINAL ROLL 18/19'] = final['TAX CLASS AS OF FINAL ROLL 18/19']\
        .map({'2A': 2, '1': 1, '4': 4})

    return final


def get_some_information_from_data(df):
    print(df.isna().sum())
    print(f'Train data has {df.shape[0]} rows and {df.shape[1]} colummns')
    plot_statistic(df)


def load_data_from_excel():
    df1 = pd.read_excel("Data/2019/2019_bronx.xlsx")
    df2 = pd.read_excel("Data/2019/2019_brooklyn.xlsx")
    df3 = pd.read_excel("Data/2019/2019_manhattan.xlsx")
    df4 = pd.read_excel("Data/2019/2019_queens.xlsx")
    df5 = pd.read_excel("Data/2019/2019_statenisland.xlsx")
    return df1, df2, df3, df4, df5


def feature_engineering(df1, df2, df3, df4, df5):
    df1, df2, df3, df4, df5 = add_area_column(df1, df2, df3, df4, df5)
    df = data_frame_concat(df1, df2, df3, df4, df5)
    df = change_columns_name(df)
    df['FINAL_ADDRESS'] = df['ADDRESS'].apply(lambda x: x + " NYC")

    return df


def add_area_column(df1, df2, df3, df4, df5):
    df1['AREA'] = 'bronx'
    df2['AREA'] = 'brooklyn'
    df3['AREA'] = 'manhattan'
    df4['AREA'] = 'queens'
    df5['AREA'] = 'statenisland'

    return df1, df2, df3, df4, df5


def data_frame_concat(df1, df2, df3, df4, df5):
    frames = [df1, df2, df3, df4, df5]
    df = pd.concat(frames)
    return df


def change_columns_name(df):
    df.columns = ['BOROUGH',
                  'NEIGHBORHOOD',
                  'BUILDING CLASS CATEGORY',
                  'TAX CLASS AS OF FINAL ROLL 18/19',
                  'BLOCK',
                  'LOT',
                  'EASE-MENT',
                  'BUILDING CLASS AS OF FINAL ROLL 18/19',
                  'ADDRESS',
                  'APARTMENT NUMBER',
                  'ZIP CODE',
                  'RESIDENTIAL UNITS',
                  'COMMERCIAL UNITS',
                  'TOTAL UNITS',
                  'LAND SQUARE FEET',
                  'GROSS SQUARE FEET',
                  'YEAR BUILT',
                  'TAX CLASS AT TIME OF SALE',
                  'BUILDING CLASS AT TIME OF SALE',
                  'SALE PRICE',
                  'SALE DATE',
                  'AREA']
    return df


def remove_outliers(df):
    df = df.drop(['EASE-MENT', 'APARTMENT NUMBER'], axis=1)

    df = df[df['SALE PRICE'] < 3e6]
    df = df[df['SALE PRICE'] > 5000]

    df = df[df['RESIDENTIAL UNITS'] < 10.0]

    df = df[df['COMMERCIAL UNITS'] < 5.0]

    df = df[df['YEAR BUILT'] > 1860]
    df = df[df['YEAR BUILT'] <= 2020]

    df = df[df['GROSS SQUARE FEET'] < 4600]
    df = df[df['GROSS SQUARE FEET'] > 50]

    df = df[df['LAND SQUARE FEET'] > 200]
    df = df[df['LAND SQUARE FEET'] < 7000]

    llimit = np.percentile(df['ZIP CODE'], 1)
    df = df[df['ZIP CODE'] > llimit]

    df = df[df['TOTAL UNITS'] < 5]
    df = df[df['TOTAL UNITS'] > 0]

    df = df[df['LOT'] < 250]

    return df


def print_graph_date_vs_price(df):
    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(10, 10))
    # Add x-axis and y-axis
    ax.bar(df['SALE DATE'].values,
           df['SALE PRICE'].values,
           color='purple')
    # Set title and labels for axes
    ax.set(xlabel="Date",
           ylabel="Counts",
           title="Counts per date")
    # Rotate tick marks on x-axis
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.show()


def plot_statistic(df):
    describe_df = df.describe()
    print(describe_df)


def plot_some_graph(df):
    int_level = df['AREA'].value_counts()

    plt.figure(figsize=(8, 4))
    sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('BOROUGH', fontsize=12)
    plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(range(df.shape[0]), df['SALE PRICE'].values)
    plt.xlabel('index', fontsize=12)
    plt.ylabel('price', fontsize=12)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.distplot(df['SALE PRICE'].values, bins=50, kde=True)
    plt.xlabel('price', fontsize=12)
    plt.show()

    int_level = df['TAX CLASS AS OF FINAL ROLL 18/19'].value_counts()

    plt.figure(figsize=(8, 4))
    sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[4])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('TAX CLASS AS OF FINAL ROLL 18/19', fontsize=12)
    plt.show()

    uniqueValues = df['RESIDENTIAL UNITS'].unique()
    print('Unique elements in column "RESIDENTIAL UNITS" ')
    print(np.sort(uniqueValues))

    cnt_srs = df['RESIDENTIAL UNITS'].value_counts()

    plt.figure(figsize=(8, 4))
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[0])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('RESIDENTIAL UNITS', fontsize=12)
    plt.show()

    uniqueValues = df['COMMERCIAL UNITS'].unique()
    print('Unique elements in column "COMMERCIAL UNITS" ')
    print(np.sort(uniqueValues))

    cnt_srs = df['COMMERCIAL UNITS'].value_counts()

    plt.figure(figsize=(8, 4))
    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[2])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('COMMERCIAL UNITS', fontsize=12)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.distplot(df['YEAR BUILT'].values, bins=50, kde=True)
    plt.xlabel('YEAR BUILT', fontsize=12)
    plt.show()

    uniqueValues = df['YEAR BUILT'].unique()
    print('Unique elements in column "YEAR BUILT" ')
    print(np.sort(uniqueValues))

    plt.figure(figsize=(8, 6))
    sns.distplot(df['GROSS SQUARE FEET'].values, bins=50, kde=True)
    plt.xlabel('GROSS SQUARE FEET', fontsize=12)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.distplot(df['LAND SQUARE FEET'].values, bins=50, kde=True)
    plt.xlabel('LAND SQUARE FEET', fontsize=12)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.distplot(df['BLOCK'].values, bins=50, kde=True)
    plt.xlabel('BLOCK', fontsize=12)
    plt.show()

    uniqueValues = df['BLOCK'].unique()
    print('Unique elements in column "BLOCK" ')
    print(np.size(np.sort(uniqueValues)))

    plt.figure(figsize=(8, 6))
    sns.distplot(df['ZIP CODE'].values, bins=50, kde=True)
    plt.xlabel('ZIP CODE', fontsize=12)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.distplot(df['TOTAL UNITS'].values, bins=50, kde=True)
    plt.xlabel('TOTAL UNITS', fontsize=12)
    plt.show()

    print_graph_date_vs_price(df)

    plt.figure(figsize=(8, 6))
    sns.distplot(df['LOT'].values, bins=50, kde=True)
    plt.xlabel('LOT', fontsize=12)
    plt.show()

    int_level = df['BUILDING CLASS CATEGORY'].value_counts()

    plt.figure(figsize=(8,4))
    sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[9])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('BUILDING CLASS CATEGORY', fontsize=12)
    plt.show()

    cnt_srs = df['BUILDING CLASS CATEGORY'].value_counts()
    print(cnt_srs)

    int_level = df['TAX CLASS AT TIME OF SALE'].value_counts()

    plt.figure(figsize=(8, 4))
    sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[9])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('TAX CLASS AT TIME OF SALE', fontsize=12)
    plt.show()

    cnt_srs = df['TAX CLASS AT TIME OF SALE'].value_counts()
    print(cnt_srs)

    int_level = df['BUILDING CLASS AT TIME OF SALE'].value_counts()

    plt.figure(figsize=(8, 4))
    sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[9])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('BUILDING CLASS AT TIME OF SALE', fontsize=12)
    plt.show()

    cnt_srs = df['BUILDING CLASS AT TIME OF SALE'].value_counts()
    print(cnt_srs)


def complete_missing_values(df: pd.DataFrame):
    df = df.dropna(how='any', axis=0)
    return df


def get_lat(address):
    geolocator = Nominatim(user_agent="real estate ML project!")
    location = geolocator.geocode(address + " NYC")
    if location is not None:
        return location.latitude
    else:
        return 'NF'


def get_long(address):
    geolocator = Nominatim(user_agent="real estate ML project!")
    location = geolocator.geocode(address + " NYC")
    if location is not None:
        return location.longitude
    else:
        return 'NF'


# def load_data(data_path: str) -> pd.DataFrame:
# TODO: activate load_excel_from_dir for each directory in the Data folder
def load_excel_from_dir(path_to_dir: str) -> pd.DataFrame:
    excel_files_list = list(filter(lambda file_name: file_name.endswith(EXCEL_EXTENSION), os.listdir(path_to_dir)))
    df_list = []
    for file in excel_files_list:
        df_list.append(pd.read_excel(os.path.join(path_to_dir, file)))
    x = pd.concat(df_list)
    return x


if __name__ == '__main__':
    main()
