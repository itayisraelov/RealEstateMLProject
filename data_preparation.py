import pandas as pd
import numpy as np


def main():
    df1 = pd.read_excel("2019_bronx.xlsx")
    df1['AREA'] = 'bronx'
    df2 = pd.read_excel("2019_brooklyn.xlsx")
    df2['AREA'] = 'brooklyn'
    df3 = pd.read_excel("2019_manhattan.xlsx")
    df3['AREA'] = 'manhattan'
    df4 = pd.read_excel("2019_queens.xlsx")
    df4['AREA'] = 'queens'
    df5 = pd.read_excel("2019_statenisland.xlsx")
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








if __name__ == '__main__':
    main()