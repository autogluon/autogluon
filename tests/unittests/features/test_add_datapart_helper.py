import pandas as pd
import numpy as np
from autogluon.utils.tabular.features.add_datepart_helper import *
import pytest
import itertools

date_col = 'dates'
dates =  ['02/03/2017T12:32:45', '02/04/2017T14:32:05', '02/05/2017T23:44:42']

def test_add_datepart():
    df = pd.DataFrame({date_col: dates})

    expected_new_cols = [
        'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
        'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'
    ]
    expected_new_cols = [ date_col+col for col in expected_new_cols ]

    new_data = expand_date_sequential(df[date_col]) # inplace
    for col in expected_new_cols:
        assert col in new_data.columns


def test_expand_date_sequential():
    df = pd.DataFrame({date_col: dates})

    expected_new_cols = [
        'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
        'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'
    ]
    expected_new_cols = [ date_col+col for col in expected_new_cols ]

    new_data = expand_date_sequential(df[date_col]) # inplace
    for col in expected_new_cols:
        assert col in new_data.columns

def test_expand_date_cyclic():
    df = pd.DataFrame({date_col: dates})

    new_cols = ['weekday', 'day_month', 'month_year', 'day_year', 'hour', 'clock_hour', 'minute', 'second']
    cyclizations = ['sin', 'cos']
    expected_new_cols = [ f'{date_col}_{col}_{fn}' for col, fn in itertools.product(new_cols, cyclizations) ]

    new_data = add_cyclic_datepart(df[date_col], drop=True, time=True)

    for col in expected_new_cols:
        assert col in new_data.columns