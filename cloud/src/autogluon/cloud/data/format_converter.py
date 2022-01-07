import os
import pandas as pd

from abc import ABC, abstractmethod
from typing import Union


class FormatConverter(ABC):

    @property
    @abstractmethod
    def ext(self) -> str:
        pass

    @abstractmethod
    def convert(self, data: Union[str, pd.DataFrame], output_path: str,) -> str:
        pass


class CSVConverter(FormatConverter):

    @property
    def ext(self) -> str:
        return 'csv'

    def convert(self, data: Union[str, pd.DataFrame], output_path: str, filename: str) -> str:
        if isinstance(data, pd.DataFrame):
            path = os.path.join(output_path, f'{filename}.{self.ext}')
            data.to_csv(path, index=None)
            return path
        elif type(data) == str:
            data = os.path.expanduser(data)
            if os.path.isfile(data):
                if data.endswith('parquet') or data.endswith('pq'):
                    df = pd.read_parquet(data)
                    return self.convert(df, output_path, filename)
                elif data.endswith('csv') or data.endswith('tsv'):
                    return data
                else:
                    raise ValueError(f'{data} type is not supported.')
            else:
                raise ValueError(f'{data} type is not supported.')
        else:
            raise ValueError(f'{data} is not supported.')


class PauquetConverter(FormatConverter):

    @property
    def ext(self) -> str:
        return 'parquet'

    def convert(self, data: Union[str, pd.DataFrame], output_path: str, filename: str) -> str:
        if isinstance(data, pd.DataFrame):
            path = os.path.join(output_path, f'{filename}.{self.ext}')
            data.to_parquet(path)
            return path
        elif type(data) == str:
            data = os.path.expanduser(data)
            if os.path.isfile(data):
                if data.endswith('parquet') or data.endswith('pq'):
                    return data
                elif data.endswith('csv') or data.endswith('tsv'):
                    df = pd.read_csv(data)
                    return self.convert(df, output_path, filename)
                else:
                    raise ValueError(f'{data} type is not supported.')
            else:
                raise ValueError(f'{data} type is not supported.')
        else:
            raise ValueError(f'{data} is not supported.')


class FormatConverterFactory():

    @staticmethod
    def get_converter(converter_type: str) -> FormatConverter:
        assert converter_type in ['csv', 'parquet'], f'{converter_type} not supported'
        if converter_type == 'csv':
            return CSVConverter()
        elif converter_type == 'parquet':
            return PauquetConverter()
