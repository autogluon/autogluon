import datetime as dt

from fastai.tabular.transform import *

from tabular.sandbox.swallows.processors.constants import *
from tabular.sandbox.swallows.processors.preprocessor import AbstractPreprocessor

EPOCH = dt.date(1899, 12, 30)


class Stage3DateFeaturesProcessor(AbstractPreprocessor):

    def __init__(self):
        self.name = "Stage3DateFeaturesProcessor"

    def run(self, context, df):
        df = df.copy()

        df['create_date_dt'] = df['create_date'].map(Stage3DateFeaturesProcessor.to_date)
        df['assigned_date_dt'] = df['assigned_date'].map(Stage3DateFeaturesProcessor.to_date)

        # Time between create and assign; NOTE: this can happen: assinged date < create_date
        df['create_to_assigned_lag'] = np.abs((df['assigned_date_dt'] - df['create_date_dt']).map(lambda x: x.days))
        df['create_to_assigned_lag_log'] = np.log1p(df['create_to_assigned_lag'])
        df['create_to_assigned_lag_log_bin'] = np.floor(df['create_to_assigned_lag_log']).astype(np.int)

        add_datepart(df, 'assigned_date_dt', 'assigned_date_')
        add_datepart(df, 'create_date_dt', 'create_date_')

        return df

    @staticmethod
    def to_date(days):
        # See why https://stackoverflow.com/questions/24853480/python-returns-5-digit-timestamp-what-is-this
        return EPOCH + dt.timedelta(days=days)
