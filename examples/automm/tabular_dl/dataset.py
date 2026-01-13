import abc
import os

import pandas as pd

from autogluon.multimodal.constants import (
    ACC,
    BINARY,
    CATEGORICAL,
    MULTICLASS,
    NUMERICAL,
    REGRESSION,
    RMSE,
)
from autogluon.multimodal.utils import download


# TODO: release the auto_mm_bench package or reuse the huggingface datasets API
class BaseTabularDataset(abc.ABC):
    @property
    @abc.abstractmethod
    def data(self):
        pass

    @property
    @abc.abstractmethod
    def label_column(self):
        pass

    @property
    @abc.abstractmethod
    def label_type(self):
        pass

    @property
    @abc.abstractmethod
    def metric(self):
        pass

    @property
    @abc.abstractmethod
    def problem_type(self):
        pass


class AdultTabularDataset(BaseTabularDataset):
    _INFO = {
        "train": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/adult/train.csv",
            "sha1sum": "7fca6a419cff0f8504f083f7534119956452a337",
        },
        "val": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/adult/val.csv",
            "sha1sum": "80fd57b2848966c8e14f5a96a97f1bfcd41ab42c",
        },
        "test": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/adult/test.csv",
            "sha1sum": "2ca8470c2514ba147da593882c76f956681e304a",
        },
    }

    def __init__(self, split="train", path="./dataset/"):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Unsupported split {split}. Split must be one of train, val, or test."
        self._split = split
        self._path = os.path.join(path, "adult", f"{split}.csv")
        download(self._INFO[split]["url"], path=self._path, sha1_hash=self._INFO[split]["sha1sum"])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @property
    def label_column(self):
        return "target"

    @property
    def label_type(self):
        return CATEGORICAL

    @property
    def metric(self):
        return ACC

    @property
    def problem_type(self):
        return BINARY


class AloiTabularDataset(BaseTabularDataset):
    _INFO = {
        "train": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/aloi/train.csv",
            "sha1sum": "62f8457a455506d83db0c671fe9d271d376ecb22",
        },
        "val": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/aloi/val.csv",
            "sha1sum": "2e0e70dd710a441ac130a7b29a420acac26fbef4",
        },
        "test": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/aloi/test.csv",
            "sha1sum": "3f9311df8b1a8cb0c0d9b51ad32e274e7f5eb36b",
        },
    }

    def __init__(self, split="train", path="./dataset/"):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Unsupported split {split}. Split must be one of train, val, or test."
        self._split = split
        self._path = os.path.join(path, "aloi", f"{split}.csv")
        download(self._INFO[split]["url"], path=self._path, sha1_hash=self._INFO[split]["sha1sum"])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @property
    def label_column(self):
        return "target"

    @property
    def label_type(self):
        return CATEGORICAL

    @property
    def metric(self):
        return ACC

    @property
    def problem_type(self):
        return MULTICLASS


class CaliforniaHousingTabularDataset(BaseTabularDataset):
    _INFO = {
        "train": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/california_housing/train.csv",
            "sha1sum": "2f8bd84e8665859ea09ec7dfc75540357e0e8b30",
        },
        "val": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/california_housing/val.csv",
            "sha1sum": "09b165f96d1d53062dc4b8da1f6257b98a320acc",
        },
        "test": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/california_housing/test.csv",
            "sha1sum": "434e092996e94f4067145db7716a7a95849759ec",
        },
    }

    def __init__(self, split="train", path="./dataset/"):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Unsupported split {split}. Split must be one of train, val, or test."
        self._split = split
        self._path = os.path.join(path, "california_housing", f"{split}.csv")
        download(self._INFO[split]["url"], path=self._path, sha1_hash=self._INFO[split]["sha1sum"])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @property
    def label_column(self):
        return "target"

    @property
    def label_type(self):
        return NUMERICAL

    @property
    def metric(self):
        return RMSE

    @property
    def problem_type(self):
        return REGRESSION


class CovtypeTabularDataset(BaseTabularDataset):
    _INFO = {
        "train": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/covtype/train.csv",
            "sha1sum": "7cd56fed5f667dcebe6fb3eb096bc0166636bc5d",
        },
        "val": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/covtype/val.csv",
            "sha1sum": "0e32c887a586963f4e7cb62d16964ec8dd57cf08",
        },
        "test": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/covtype/test.csv",
            "sha1sum": "35883acd6db686e148338bb7c6f0c46df0039574",
        },
    }

    def __init__(self, split="train", path="./dataset/"):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Unsupported split {split}. Split must be one of train, val, or test."
        self._split = split
        self._path = os.path.join(path, "covtype", f"{split}.csv")
        download(self._INFO[split]["url"], path=self._path, sha1_hash=self._INFO[split]["sha1sum"])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @property
    def label_column(self):
        return "target"

    @property
    def label_type(self):
        return CATEGORICAL

    @property
    def metric(self):
        return ACC

    @property
    def problem_type(self):
        return MULTICLASS


class EpsilonTabularDataset(BaseTabularDataset):
    _INFO = {
        "train": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular/epsilon/train.csv",
            "sha1sum": "8444901bdb20d42359b85ca076eff7f16a34b94c",
        },
        "val": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular/epsilon/val.csv",
            "sha1sum": "9d607e0db43979d3d9a6034dc7603ef09934b5c8",
        },
        "test": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular/epsilon/test.csv",
            "sha1sum": "0a33633875a87a8c9f316e78d225ddfb41c54718",
        },
    }

    def __init__(self, split="train", path="./dataset/"):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Unsupported split {split}. Split must be one of train, val, or test."
        self._split = split
        self._path = os.path.join(path, "epsilon", f"{split}.csv")
        download(self._INFO[split]["url"], path=self._path, sha1_hash=self._INFO[split]["sha1sum"])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @property
    def label_column(self):
        return "target"

    @property
    def label_type(self):
        return CATEGORICAL

    @property
    def metric(self):
        return ACC

    @property
    def problem_type(self):
        return BINARY


class HelenaTabularDataset(BaseTabularDataset):
    _INFO = {
        "train": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/helena/train.csv",
            "sha1sum": "61f290ee851695715662177b5726055cc7f8bccb",
        },
        "val": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/helena/val.csv",
            "sha1sum": "67d01556af5f78015de1151e3a3c81f71de6bc11",
        },
        "test": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/helena/test.csv",
            "sha1sum": "c55709b71ce79051e5f580e9edb73be81d41bf92",
        },
    }

    def __init__(self, split="train", path="./dataset/"):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Unsupported split {split}. Split must be one of train, val, or test."
        self._split = split
        self._path = os.path.join(path, "helena", f"{split}.csv")
        download(self._INFO[split]["url"], path=self._path, sha1_hash=self._INFO[split]["sha1sum"])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @property
    def label_column(self):
        return "target"

    @property
    def label_type(self):
        return CATEGORICAL

    @property
    def metric(self):
        return ACC

    @property
    def problem_type(self):
        return MULTICLASS


class HiggsSmallTabularDataset(BaseTabularDataset):
    _INFO = {
        "train": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/higgs_small/train.csv",
            "sha1sum": "65554dceee2c6647f348153412c099891c3b0516",
        },
        "val": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/higgs_small/val.csv",
            "sha1sum": "63ab6b0ac6730ea63c18d106c6eb17f5874042f8",
        },
        "test": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/higgs_small/test.csv",
            "sha1sum": "dd372975aae92ffcc3a0335299382f7ff1d889bc",
        },
    }

    def __init__(self, split="train", path="./dataset/"):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Unsupported split {split}. Split must be one of train, val, or test."
        self._split = split
        self._path = os.path.join(path, "higgs_small", f"{split}.csv")
        download(self._INFO[split]["url"], path=self._path, sha1_hash=self._INFO[split]["sha1sum"])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @property
    def label_column(self):
        return "target"

    @property
    def label_type(self):
        return CATEGORICAL

    @property
    def metric(self):
        return ACC

    @property
    def problem_type(self):
        return BINARY


class JannisTabularDataset(BaseTabularDataset):
    _INFO = {
        "train": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/jannis/train.csv",
            "sha1sum": "6b2cb95c8858aac965908e6973d8728ce13152c5",
        },
        "val": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/jannis/val.csv",
            "sha1sum": "17332c6155d60aeb30ddc201f9da8dc93fa05584",
        },
        "test": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/jannis/test.csv",
            "sha1sum": "c27654b99794de445bc4053ccb052800886d0d0a",
        },
    }

    def __init__(self, split="train", path="./dataset/"):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Unsupported split {split}. Split must be one of train, val, or test."
        self._split = split
        self._path = os.path.join(path, "jannis", f"{split}.csv")
        download(self._INFO[split]["url"], path=self._path, sha1_hash=self._INFO[split]["sha1sum"])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @property
    def label_column(self):
        return "target"

    @property
    def label_type(self):
        return CATEGORICAL

    @property
    def metric(self):
        return ACC

    @property
    def problem_type(self):
        return MULTICLASS


class MicrosoftTabularDataset(BaseTabularDataset):
    _INFO = {
        "train": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/microsoft/train.csv",
            "sha1sum": "40f773e920798759c59d0afc6b83112389953a0a",
        },
        "val": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/microsoft/val.csv",
            "sha1sum": "351b43ce2eddb9af3ce38c5404e6c2eb4907392f",
        },
        "test": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/microsoft/test.csv",
            "sha1sum": "03a10cb4b34010a41c87e8268f63cfb08ec93711",
        },
    }

    def __init__(self, split="train", path="./dataset/"):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Unsupported split {split}. Split must be one of train, val, or test."
        self._split = split
        self._path = os.path.join(path, "microsoft", f"{split}.csv")
        download(self._INFO[split]["url"], path=self._path, sha1_hash=self._INFO[split]["sha1sum"])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @property
    def label_column(self):
        return "target"

    @property
    def label_type(self):
        return NUMERICAL

    @property
    def metric(self):
        return RMSE

    @property
    def problem_type(self):
        return REGRESSION


class YahooTabularDataset(BaseTabularDataset):
    _INFO = {
        "train": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/yahoo/train.csv",
            "sha1sum": "9444919528cd4f8429b049e98e67d95ca743b607",
        },
        "val": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/yahoo/val.csv",
            "sha1sum": "627ba212d51719449453e24aaa7a14435ca97d8b",
        },
        "test": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/yahoo/test.csv",
            "sha1sum": "ad8acb167c1d125c47380e57ac50de120072af23",
        },
    }

    def __init__(self, split="train", path="./dataset/"):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Unsupported split {split}. Split must be one of train, val, or test."
        self._split = split
        self._path = os.path.join(path, "yahoo", f"{split}.csv")
        download(self._INFO[split]["url"], path=self._path, sha1_hash=self._INFO[split]["sha1sum"])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @property
    def label_column(self):
        return "target"

    @property
    def label_type(self):
        return NUMERICAL

    @property
    def metric(self):
        return RMSE

    @property
    def problem_type(self):
        return REGRESSION


class YearTabularDataset(BaseTabularDataset):
    _INFO = {
        "train": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/year/train.csv",
            "sha1sum": "52a5b79a14a0fd69c94105f3212d9728dfc658ed",
        },
        "val": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/year/val.csv",
            "sha1sum": "f5469142509a75c8ba2915fda74efe1fea221058",
        },
        "test": {
            "url": "https://autogluon.s3.us-west-2.amazonaws.com/datasets/tabular_example/year/test.csv",
            "sha1sum": "d5cf1b15c255a8a6ac0ab94f83373266944a7675",
        },
    }

    def __init__(self, split="train", path="./dataset/"):
        assert split in [
            "train",
            "val",
            "test",
        ], f"Unsupported split {split}. Split must be one of train, val, or test."
        self._split = split
        self._path = os.path.join(path, "year", f"{split}.csv")
        download(self._INFO[split]["url"], path=self._path, sha1_hash=self._INFO[split]["sha1sum"])
        self._data = pd.read_csv(self._path)

    @property
    def data(self):
        return self._data

    @property
    def label_column(self):
        return "target"

    @property
    def label_type(self):
        return NUMERICAL

    @property
    def metric(self):
        return RMSE

    @property
    def problem_type(self):
        return REGRESSION
