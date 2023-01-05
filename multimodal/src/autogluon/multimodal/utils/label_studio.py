import json
import os
from enum import Enum

import pandas as pd


class TaskType(Enum):
    """
    Currently supported task type
    """

    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    CUSTOMIZE = "customize"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_SUMMARIZATION = "text_summarization"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"


# preset data and label columns of some given label-studio template (except customized)
columns_template = {
    "image_classification": {
        "csv": {"data_columns": ["image"], "label_columns": ["choice"]},
        "json": {"data_columns": ["image"], "label_columns": ["value.choices"]},
        "json_min": {"data_columns": ["image"], "label_columns": ["choice"]},
    },
    "object_detection": {
        "csv": {"data_columns": ["image"], "label_columns": ["label"]},
        "json": {
            "data_columns": ["image"],
            "label_columns": [
                "original_width",
                "original_height",
                "value.x",
                "value.y",
                "value.width",
                "value.height",
                "value.rotation",
                "value.rectanglelabels",
            ],
        },
        "json_min": {"data_columns": ["image"], "label_columns": ["label"]},
    },
    "named_entity_recognition": {
        "csv": {"data_columns": ["text"], "label_columns": ["label"]},
        "json": {
            "data_columns": ["text"],
            "label_columns": ["value.start", "value.end", "value.text", "value.labels"],
        },
        "json_min": {"data_columns": ["text"], "label_columns": ["label"]},
    },
}


def read_from_labelstudio_csv(path, data_columns, label_columns):
    """
    read data from exported .csv files and return an Dataframe

    params:
    - path: str: the path of the exported file
    - data_columns: list[str]: the key/column names of the data
    - label_columns: list[str]: the key/column names of the label
    """
    if not os.path.exists(path):
        raise OSError("annotation file path not exists.")
    labelstudio_csv = pd.read_csv(path)
    df = pd.DataFrame()
    columns = labelstudio_csv.columns
    # extracting columns for given data and label column names
    for col in data_columns:
        if col in columns:
            df[col] = labelstudio_csv[col]
        else:
            print("skip '{}' for not in the export data columns".format(col))
    for col in label_columns:
        if col in columns:
            df[col] = labelstudio_csv[col]
        else:
            print("skip '{}' for not in the export label columns".format(col))
    return df


def read_from_labelstudio_json(path, data_columns, label_columns):
    """
    read data from exported .json files and return an Dataframe
    (include the JSON and JSON-MIN files from Label-Studio)

    params:
    - path: str: the path of the exported file
    - data_columns: list[str]: the key/column names of the data
    - label_columns: list[str]: the key/column names of the label
    """
    if not os.path.exists(path):
        raise OSError("annotation file path not exists.")

    with open(path, mode="r") as fp:
        label_studio_json = json.load(fp)
        if len(label_studio_json) == 0:
            raise ValueError("ERROR: empty export file")

        if "annotations" in label_studio_json[0]:
            # the file is export through JSON
            # JSON file parsing is available to specify nested data
            annotation = pd.json_normalize(
                data=label_studio_json, record_path=["annotations", "result"], meta=["data"]
            )
            data = pd.DataFrame(annotation["data"].values.tolist())

            # annotation: labeling content
            # data: data content/url/...
            df = pd.DataFrame()

            columns = data.columns
            for col in data_columns:
                if col in columns:
                    df[col] = data[col]
                else:
                    print("skip '{}' for not in the export data columns".format(col))

            columns = annotation.columns
            for col in label_columns:
                if col in columns:
                    df[col] = annotation[col]
                else:
                    print("skip '{}' for not in the export label columns".format(col))

            return df

        else:
            # the file is export through JSON-MIN

            df = pd.DataFrame()
            annotation_table = pd.json_normalize(data=label_studio_json)

            for col in data_columns:
                if col in annotation_table:
                    df[col] = annotation_table[col]
                else:
                    print("skip '{}' for not in the export data columns".format(col))

            for col in label_columns:
                if col in annotation_table:
                    df[col] = annotation_table[col]
                else:
                    print("skip '{}' for not in the export label columns".format(col))

            return df


def get_dataframes_by_path(path, data_columns, label_columns):
    """
    get the data frames by path and given data, label column names,
    and return the Dataframe

    params:
    - path: str: the path of the exported file
    - data_columns: list[str]: the key/column names of the data
    - label_columns: list[str]: the key/column names of the label
    """

    # get file extension through os
    _, file_extension = os.path.splitext(path)
    file_extension = file_extension[1:]

    if file_extension == "csv":
        return read_from_labelstudio_csv(path, data_columns, label_columns)
    elif file_extension == "json" or file_extension == "json_min":
        return read_from_labelstudio_json(path, data_columns, label_columns)
    else:
        raise OSError("current file extension {} is not supported.".format(file_extension))


class LabelStudioReader:
    """
    a tool that transfer label-studio export file to the dataframe for autogluon training. Docs available at
    https://github.com/autogluon/autogluon/tree/master/examples/automm/label_studio/label_studio_export_reader
    """

    def __init__(self, host=None):
        self.default_host = "http://localhost:8080" if not host else host
        print(
            "NOTE: the default label-studio host is {},if you want to get data from an running label-studio url, "
            "please set 'ls_host_on' to True".format(self.default_host)
        )
        self.templates = columns_template

    def set_labelstudio_host(self, host):
        if host:
            self.default_host = host
            print("set label-studio default host to {}".format(host))

    def get_columns_by_type(self, type, format):
        """
        for tasks that labeled from a given Label-Studio Template with the names
        of the template not modified, the data_columns and label_columns can be
        achieved through the preset template.
        params:
        - type: Enum of the template type(e.g: image classification).See class TaskType.
        - format: str: file export format: (csv, json, json_min)
        """
        if type.value != "customize":
            data_columns = self.templates[type.value][format]["data_columns"]
            label_columns = self.templates[type.value][format]["label_columns"]
            return data_columns, label_columns
        else:
            return [], []

    def process_data_str(self, s: str, ls_online):
        """
        used for lambda expression of the data str. See the detailed documentation
        params:
        - s: str, the origin data str
        - ls_online: boolean: whether the label-studio host should be provided
        """
        if ls_online:
            if s.startswith("/data/"):
                # label-studio imported data.
                # data are accessible when label-studio host is on
                # data can be addressed through a temporal ls-host-based URL
                return self.default_host + s
            else:
                # url, text content,...
                # need no further process
                return s

        else:
            local_storage_prefix = "/data/local-files/?d="
            upload_prefix = "/data/upload"
            if s.startswith(local_storage_prefix):
                # data imported from Label-Studio local storage
                # changed to the local path of the file
                return s[len(local_storage_prefix) :]
            elif s.startswith(upload_prefix):
                # the uploaded file can not be accessed
                print("Warning: cannot read {} with the label-studio host off.".format(s))
                return s
            # TODO: add s3, gcd, redis support
            else:
                return s

    def from_image_classification(self, path, ls_host_on, data_columns=None, label_columns=None):
        """
        data: image files
        process export file from label-studio image classification template,
        return the overall and label Dataframe for Autogluon input
        param:
        - path: str: the local path of exported file
        - ls_host_on: boolean: if the label-studio host is needed
        - data_columns: list[str](Optional) key or column names of data
        - label_columns: list[str](Optional) key or column names of label
        """

        if not os.path.exists(path):
            raise OSError("annotation file path not exists.")

        # get file extension with os, distinguish JSON and JSON-MIN
        _, file_extension = os.path.splitext(path)
        file_extension = file_extension[1:]
        if file_extension == "json":
            with open(path, mode="r") as fp:
                json_content = json.load(fp)
                if "annotations" not in json_content[0]:
                    file_extension = "json_min"

        # get the data and label columns through default image classification template
        default_data, default_label = self.get_columns_by_type(TaskType.IMAGE_CLASSIFICATION, format=file_extension)
        # use user preset if there's any
        data = data_columns if data_columns else default_data
        label = label_columns if label_columns else default_label

        df = get_dataframes_by_path(path, data, label)

        columns = df.columns
        for col in data:
            if col in columns:
                # process the data file content/url according to ls_host_on
                df[col] = df[col].apply(lambda s: self.process_data_str(s, ls_host_on))
            else:
                print("skip '{}' for not in the data column names.".format(col))

        return df, df[label]

    def from_named_entity_recognition(self, path, data_columns=None, label_columns=None):
        """
        data: text
        process export file from label-studio NER template,
        return the overall and label Dataframe for Autogluon input
        param:
        - path: str: the local path of exported file
        - data_columns: list[str](Optional) key or column names of data
        - label_columns: list[str](Optional) key or column names of label
        """

        if not os.path.exists(path):
            raise OSError("annotation file path not exists.")

        # get file extension with os,distinguish JSON and JSON-MIN
        _, file_extension = os.path.splitext(path)
        file_extension = file_extension[1:]
        if file_extension == "json":
            with open(path, mode="r") as fp:
                json_content = json.load(fp)
                if "annotations" not in json_content[0]:
                    file_extension = "json_min"

        # get the data and label columns through default NER template
        default_data, default_label = self.get_columns_by_type(
            TaskType.NAMED_ENTITY_RECOGNITION, format=file_extension
        )
        data = data_columns if data_columns else default_data
        label = label_columns if label_columns else default_label

        df = get_dataframes_by_path(path, data, label)
        return df, df[label]

    def from_customize(self, path, ls_host_on, data_columns, label_columns):
        """
        process export file from customized template,
        return the overall and label Dataframe for Autogluon input
        param:
        - path: str: the local path of exported file
        - ls_host_on: boolean: if the label-studio host is needed
        - data_columns: list[str](Optional) key or column names of data
        - label_columns: list[str](Optional) key or column names of label
        """

        if not os.path.exists(path):
            raise OSError("annotation file path not exists.")

        _, file_extension = os.path.splitext(path)
        file_extension = file_extension[1:]
        if file_extension == "json":
            with open(path, mode="r") as fp:
                json_content = json.load(fp)
                if "annotations" not in json_content[0]:
                    file_extension = "json_min"

        data, label = data_columns, label_columns

        df = get_dataframes_by_path(path, data, label)

        columns = df.columns
        for col in data:
            if col in columns:
                df[col] = df[col].apply(lambda s: self.process_data_str(s, ls_host_on))
            else:
                print("skip '{}' for not in the data column names.".format(col))

        return df, df[label]
