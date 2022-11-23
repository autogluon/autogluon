import os
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


class FormatConverter(ABC):
    @property
    @abstractmethod
    def ext(self) -> str:
        """
        File extension of this FormatConverter.
        """
        raise NotImplementedError

    @abstractmethod
    def _need_conversion(self, filename: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _save_dataframe(self, data: pd.DataFrame, path: str):
        raise NotImplementedError

    def read_file(self, filename: str) -> pd.DataFrame:
        """
        Read in file as a pandas DataFrame

        Parameters
        ----------
        filename: str
            Path to the file to read.

        Returns
        ------
        pd.DataFrame
        """
        if FormatConverter.is_parquet_file(filename):
            data = pd.read_parquet(filename)
        elif FormatConverter.is_csv_file(filename):
            data = pd.read_csv(filename)
        else:
            ext = filename.split(".")[-1]
            raise ValueError(f"{ext} file type is not supported.")
        return data

    def convert(self, data: Union[str, pd.DataFrame], output_path: str, filename: str) -> str:
        """
        Convert a tabular file to another format.
        If the file does not need conversion, will return the original path.

        Parameters
        ----------
        data: Union[str, pd.DataFrame]
            If str, path to the file to be converted.
            If pd.DataFrame, dataframe to be converted.
        output_path: str
            Path to save the converted file
        filename: str
            Filename to be saved for the converted file

        Returns
        ------
        str
            Path to the converted file. If the file does not need conversion, will return the original path.
        """
        if type(data) == str:
            data = os.path.expanduser(data)
            if os.path.isfile(data):
                if self._need_conversion(data):
                    data = self.read_file(data)
                else:
                    return data
            else:
                raise ValueError("Please provide a path to a file.")

        if isinstance(data, pd.DataFrame):
            path = os.path.join(output_path, f"{filename}.{self.ext}")
            self._save_dataframe(data, path)
            return path

        raise ValueError(f"{type(data)} is not supported.")

    @staticmethod
    def is_csv_file(filename: str) -> bool:
        """If the file is a csv file or not"""
        return filename.endswith(".csv") or filename.endswith(".tsv")

    @staticmethod
    def is_parquet_file(filename: str) -> bool:
        """If the file is a parquet file or not"""
        return filename.endswith(".parquet") or filename.endswith(".pq")


class CSVConverter(FormatConverter):
    @property
    def ext(self) -> str:
        return "csv"

    def _need_conversion(self, filename: str) -> bool:
        return not FormatConverter.is_csv_file(filename)

    def _save_dataframe(self, data: pd.DataFrame, path: str):
        data.to_csv(path, index=None)


class ParquetConverter(FormatConverter):
    @property
    def ext(self) -> str:
        return "parquet"

    def _need_conversion(self, filename: str) -> bool:
        return not FormatConverter.is_parquet_file(filename)

    def _save_dataframe(self, data: pd.DataFrame, path: str):
        data.to_parquet(path, index=None)


class FormatConverterFactory:

    __supported_converters = [CSVConverter, ParquetConverter]
    __ext_to_converter = {cls().ext: cls for cls in __supported_converters}

    @staticmethod
    def get_converter(converter_type: str) -> FormatConverter:
        """Return the corresponding converter"""
        assert converter_type in FormatConverterFactory.__ext_to_converter, f"{converter_type} not supported"
        return FormatConverterFactory.__ext_to_converter[converter_type]()
