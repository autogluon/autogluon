import pandas as pd
from sagemaker.deserializers import SimpleBaseDeserializer
import io


class PandasDeserializer(SimpleBaseDeserializer):
    """Deserialize Parquet, CSV or JSON data from an inference endpoint into a pandas dataframe."""

    def __init__(self, accept=("application/x-parquet", "text/csv", "application/json")):
        """Initialize a ``PandasDeserializer`` instance.

        Args:
            accept (union[str, tuple[str]]): The MIME type (or tuple of allowable MIME types) that
                is expected from the inference endpoint (default: ("application/x-parquet", "text/csv","application/json")).
        """
        super().__init__(accept=accept)

    def deserialize(self, stream, content_type):
        """Deserialize CSV or JSON data from an inference endpoint into a pandas dataframe.

        If the data is JSON, the data should be formatted in the 'columns' orient.
        See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html

        Args:
            stream (botocore.response.StreamingBody): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            pandas.DataFrame: The data deserialized into a pandas DataFrame.
        """
        if content_type == "application/x-parquet":
            return pd.read_parquet(io.BytesIO(stream.read()))

        if content_type == "text/csv":
            return pd.read_csv(stream)

        if content_type == "application/json":
            return pd.read_json(stream)

        raise ValueError("%s cannot read content type %s." % (__class__.__name__, content_type))
