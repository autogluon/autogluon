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

    @staticmethod
    def is_csv_file(filename: str) -> bool:
        return filename.endswith('.csv') or filename.endswith('.tsv')

    @staticmethod
    def is_parquet_file(filename: str) -> bool:
        return filename.endswith('.parquet') or filename.endswith('.pq')


class CSVConverter(FormatConverter):

    @property
    def ext(self) -> str:
        return 'csv'

    def convert(self, data: Union[str, pd.DataFrame], output_path: str, filename: str) -> str:
        if type(data) == str:
            data = os.path.expanduser(data)
            if os.path.isfile(data):
                if FormatConverter.is_parquet_file(data):
                    data = pd.read_parquet(data)
                elif FormatConverter.is_csv_file(data):
                    return data
                else:
                    ext = data.split('.')[-1]
                    raise ValueError(f'{ext} file type is not supported.')
            else:
                raise ValueError('Please provide a path to a file.')

        if isinstance(data, pd.DataFrame):
            path = os.path.join(output_path, f'{filename}.{self.ext}')
            data.to_csv(path, index=None)
            return path

        raise ValueError(f'{type(data)} is not supported.')


class ParquetConverter(FormatConverter):

    @property
    def ext(self) -> str:
        return 'parquet'

    def convert(self, data: Union[str, pd.DataFrame], output_path: str, filename: str) -> str:
        if type(data) == str:
            data = os.path.expanduser(data)
            if os.path.isfile(data):
                if FormatConverter.is_parquet_file(data):
                    return data
                elif FormatConverter.is_csv_file(data):
                    data = pd.read_csv(data)
                else:
                    ext = data.split('.')[-1]
                    raise ValueError(f'{ext} file type is not supported.')
            else:
                raise ValueError('Please provide a path to a file.')

        if isinstance(data, pd.DataFrame):
            path = os.path.join(output_path, f'{filename}.{self.ext}')
            data.to_parquet(path)
            return path

        raise ValueError(f'{type(data)} is not supported.')


class FormatConverterFactory():

    @staticmethod
    def get_converter(converter_type: str) -> FormatConverter:
        assert converter_type in ['csv', 'parquet'], f'{converter_type} not supported'
        if converter_type == 'csv':
            return CSVConverter()
        elif converter_type == 'parquet':
            return ParquetConverter()
