import pandas as pd

from sagemaker.serializers import SimpleBaseSerializer


class ParquetSerializer(SimpleBaseSerializer):
    """Serialize data to a buffer using the .parquet format."""

    def __init__(self, content_type="application/x-parquet"):
        """Initialize a ``ParquetSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "application/x-parquet").
        """
        super(ParquetSerializer, self).__init__(content_type=content_type)

    def serialize(self, data):
        """Serialize data to a buffer using the .parquet format.

        Args:
            data (object): Data to be serialized. Can be a Pandas Dataframe,
                file, or buffer.

        Returns:
            io.BytesIO: A buffer containing data serialzied in the .parquet format.
        """
        if isinstance(data, pd.DataFrame):
            return data.to_parquet()

        # files and buffers. Assumed to hold parquet-formatted data.
        if hasattr(data, "read"):
            return data.read()

        raise ValueError(f'{data} format is not supported. Please provide a DataFrame, parquet file, or buffer.')


class JsonLineSerializer(SimpleBaseSerializer):
    """Serialize data to a buffer using the .jsonl format."""

    def __init__(self, content_type="application/jsonl"):
        """Initialize a ``JsonLineSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "application/jsonl").
        """
        super(JsonLineSerializer, self).__init__(content_type=content_type)

    def serialize(self, data):
        """Serialize data to a buffer using the .jsonl format.

        Args:
            data (pd.DataFrame): Data to be serialized.

        Returns:
            io.StringIO: A buffer containing data serialzied in the .jsonl format.
        """
        if isinstance(data, pd.DataFrame):
            return data.to_json(orient='records', lines=True)

        raise ValueError(f'{data} format is not supported. Please provide a DataFrame.')
