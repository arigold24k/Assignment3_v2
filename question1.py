# from scipy.io import arff
import csv

import pandas as pd
import numpy as np
from datetime import datetime
import arff
import os

file_loc = 'mesocyclone.csv'


def main():
    try:
        data = pd.read_csv(file_loc, index_col=False)
        pd.set_option('display.max_columns', None)
        pd.reset_option('max_columns')
        pd.set_option('display.max_rows', 25)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        normalized = pd.DataFrame()
        for i in data.columns:
            # z-score normilization value - mean/ standard deviation
            normalized[i + '_N'] = (data[i] - data[i].mean()) / data[i].std()
            print(i)
        print(normalized)
        # normalized.to_csv(os.path.join(basepath, 'normalized_data.csv'))

    except Exception as e:
        print('Error: ' + str(e))


if __name__ == '__main__':
    main()
