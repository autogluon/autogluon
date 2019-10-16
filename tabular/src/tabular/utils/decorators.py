#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# importing libraries
import time
import math
from multiprocessing import Pool
import multiprocessing
import numpy as np
import pandas as pd


# decorator to calculate duration
# taken by any function.
def calculate_time(func):
    # added arguments inside the inner1,
    # if function takes any arguments,
    # can be added like this.
    def inner1(*args, **kwargs):
        # storing time before function execution
        begin = time.time()

        output = func(*args, **kwargs)

        # storing time after function execution
        end = time.time()
        print("Total time taken in : ", func.__name__, end - begin)

        return output

    return inner1


num_cores = multiprocessing.cpu_count()
num_partitions = num_cores


# decorator to calculate duration
# taken by any function.
def parallelize_df(func):
    # added arguments inside the inner1,
    # if function takes any arguments,
    # can be added like this.
    def inner1(*args, **kwargs):
        df = kwargs['df']

        # df = func(df)

        a = np.array_split(df, num_partitions)
        # del df
        with Pool(num_cores) as pool:
            df = pd.concat(pool.map(func, [a]))

        return df

    return inner1


# decorator to calculate duration
# taken by any function.
@calculate_time
def parallelize_df_2(func, df):


    # df = func(df)
    a = np.array_split(df, num_partitions)
    del df
    with Pool(num_cores) as pool:
        df = pd.concat(pool.map(func, a))

    return df


# @parallelize_df
# def f(df):
#     df['tmp'] = df['tmp'] + 1
#     return df
#
#
# if __name__ == '__main__':
#
#     data = [2*i for i in range(10000)]
#     df = pd.DataFrame(data=data, columns=['tmp'])
#
#     df2 = f(df=df)
#
#     print(df)
#     print(df2)
#


