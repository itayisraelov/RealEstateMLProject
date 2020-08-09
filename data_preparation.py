import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    df1 = pd.read_excel("Data/2019/2019_bronx.xlsx")
    df1['AREA'] = 'bronx'

    df2 = pd.read_excel("Data/2019/2019_brooklyn.xlsx")
    df2['AREA'] = 'brooklyn'

    df3 = pd.read_excel("Data/2019/2019_manhattan.xlsx")
    df3['AREA'] = 'manhattan'

    df4 = pd.read_excel("Data/2019/2019_queens.xlsx")
    df4['AREA'] = 'queens'

    df5 = pd.read_excel("Data/2019/2019_statenisland.xlsx")
    df5['AREA'] = 'statenisland'

    frames = [df1, df2, df3, df4, df5]
    df = pd.concat(frames)

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

    df.drop(df.loc[df['SALE PRICE'] == 0].index, inplace=True)
    df = df.drop(['EASE-MENT', 'APARTMENT NUMBER'], axis=1)

    df = complete_missing_values(df)

    plot_some_graph(df)

    plot_statistic(df)


def plot_statistic(df):
    describe_df = df.describe()
    print(describe_df)


def plot_some_graph(df):
    # area vs price
    # gca stands for 'get current axis'
    ax = plt.gca()
    df.plot(kind='line', x='AREA', y='SALE PRICE', ax=ax)
    # df.plot(kind='line',x='AREA',y='NEIGHBORHOOD',color='red', ax=ax)
    plt.show()

    # date vs number of transactions
    df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], infer_datetime_format=True)
    plt.clf()
    df['SALE DATE'].map(lambda d: d.month).plot(kind='hist')
    plt.show()

    # zip code vs price
    ax = plt.gca()
    df.plot(kind='line', x='ZIP CODE', y='SALE PRICE', ax=ax)
    plt.show()

    # block(tax)  vs price
    ax = plt.gca()
    df.plot(kind='line',x='BLOCK',y='SALE PRICE',ax=ax)
    plt.show()

    # total units vs price
    ax = plt.gca()
    df.plot(kind='line',x='TOTAL UNITS',y='SALE PRICE',ax=ax)
    plt.show()


def complete_missing_values(df: pd.DataFrame):
    # Complete missing entries with the most common numbers
    for col in df.columns.values:
        filler = df[col].mode()[0]
        df[col].fillna(filler, inplace=True)

    return df


if __name__ == '__main__':
    main()