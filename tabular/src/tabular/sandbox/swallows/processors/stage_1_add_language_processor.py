import multiprocessing

import numpy as np
import pandas as pd
from langdetect import detect

from tabular.sandbox.swallows.processors.preprocessor import AbstractPreprocessor


class AddDetailsLanguage(AbstractPreprocessor):

    def __init__(self):
        self.name = "AddDescriptionLanguage"
        self.c = 0

    def try_decect_lang(self, x):
        try:
            result = detect(x)
            self.c = self.c + 1
            if self.c % 10000 == 0:
                print(f'\t{self.c}')
        except:
            result = 'unknown'
        return result

    def process_chunk(self, df):
        df['lang'] = df['details'].str.strip().map(lambda x: self.try_decect_lang(x))
        return df

    def run(self, context, df):
        # TODO multiprocess

        num_splits = 4

        print('splitting target into', num_splits, 'chunks, of average size', len(df) / num_splits)
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpu_count)
        print('running pool with', cpu_count, 'workers')

        processed_chunks = pool.map(self.process_chunk, np.array_split(df, num_splits))
        df = pd.concat(processed_chunks)

        return df
