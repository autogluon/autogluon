import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from autogluon.multimodal.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.multimodal.utils import download, path_expander, path_to_bytearray_expander, protected_zip_extraction

from .utils import get_data_home_dir, get_repo_url


class PetFinderDataset:
    def __init__(
        self,
        is_bytearray=False,
    ):
        sha1sum_id = "72cb19612318bb304d4a169804f525f88dc3f0d0"
        dataset = "petfinder"
        file_name = f"{dataset}_for_unit_tests.zip"
        url = get_repo_url() + file_name
        save_path = os.path.join(get_data_home_dir(), file_name)
        self._path = os.path.join(get_data_home_dir(), dataset)
        download(
            url=url,
            path=save_path,
            sha1_hash=sha1sum_id,
        )
        protected_zip_extraction(
            save_path,
            sha1_hash=sha1sum_id,
            folder=self._path,
        )
        self._train_df = pd.read_csv(os.path.join(self._path, "train.csv"), index_col=0)
        self._test_df = pd.read_csv(os.path.join(self._path, "test.csv"), index_col=0)
        expander = path_to_bytearray_expander if is_bytearray else path_expander
        for img_col in self.image_columns:
            self._train_df[img_col] = self._train_df[img_col].apply(
                lambda ele: expander(ele, base_folder=os.path.join(self._path, "images"))
            )
            self._test_df[img_col] = self._test_df[img_col].apply(
                lambda ele: expander(ele, base_folder=os.path.join(self._path, "images"))
            )

        _, self._train_df = train_test_split(
            self._train_df,
            test_size=0.3,
            random_state=np.random.RandomState(123),
            stratify=self._train_df[self.label_columns[0]],
        )
        _, self._test_df = train_test_split(
            self._test_df,
            test_size=0.3,
            random_state=np.random.RandomState(123),
            stratify=self._test_df[self.label_columns[0]],
        )
        self._train_df.reset_index(drop=True, inplace=True)
        self._test_df.reset_index(drop=True, inplace=True)

        print(f"train sample num: {len(self._train_df)}")
        print(f"test sample num: {len(self._test_df)}")

    @property
    def path(self):
        return self._path

    @property
    def feature_columns(self):
        return [
            "Type",
            "Name",
            "Age",
            "Breed1",
            "Breed2",
            "Gender",
            "Color1",
            "Color2",
            "Color3",
            "MaturitySize",
            "FurLength",
            "Vaccinated",
            "Dewormed",
            "Sterilized",
            "Health",
            "Quantity",
            "Fee",
            "State",
            "VideoAmt",
            "Description",
            "PhotoAmt",
            "Images",
        ]

    @property
    def label_columns(self):
        return ["AdoptionSpeed"]

    @property
    def train_df(self):
        return self._train_df

    @property
    def test_df(self):
        return self._test_df

    @property
    def image_columns(self):
        return ["Images"]

    @property
    def metric(self):
        return "quadratic_kappa"

    @property
    def problem_type(self):
        return MULTICLASS


class HatefulMeMesDataset:
    def __init__(
        self,
        is_bytearray=False,
    ):
        sha1sum_id = "2aae657b786f505004ac2922b66097d60a540a58"
        dataset = "hateful_memes"
        file_name = f"{dataset}_for_unit_tests.zip"
        url = get_repo_url() + file_name
        save_path = os.path.join(get_data_home_dir(), file_name)
        self._path = os.path.join(get_data_home_dir(), dataset)
        download(
            url=url,
            path=save_path,
            sha1_hash=sha1sum_id,
        )
        protected_zip_extraction(
            save_path,
            sha1_hash=sha1sum_id,
            folder=self._path,
        )
        self._train_df = pd.read_csv(os.path.join(self._path, "train.csv"), index_col=0)
        self._test_df = pd.read_csv(os.path.join(self._path, "test.csv"), index_col=0)
        expander = path_to_bytearray_expander if is_bytearray else path_expander
        for img_col in self.image_columns:
            self._train_df[img_col] = self._train_df[img_col].apply(
                lambda ele: expander(ele, base_folder=os.path.join(self._path, "images"))
            )
            self._test_df[img_col] = self._test_df[img_col].apply(
                lambda ele: expander(ele, base_folder=os.path.join(self._path, "images"))
            )
        self._train_df.reset_index(drop=True, inplace=True)
        self._test_df.reset_index(drop=True, inplace=True)

        print(f"train sample num: {len(self._train_df)}")
        print(f"test sample num: {len(self._test_df)}")

    @property
    def path(self):
        return self._path

    @property
    def feature_columns(self):
        return ["img", "text"]

    @property
    def label_columns(self):
        return ["label"]

    @property
    def train_df(self):
        return self._train_df

    @property
    def test_df(self):
        return self._test_df

    @property
    def image_columns(self):
        return ["img"]

    @property
    def metric(self):
        return "roc_auc"

    @property
    def problem_type(self):
        return BINARY


class AEDataset:
    def __init__(
        self,
    ):
        sha1sum_id = "8c2a25555c49ef2b30545004488022465808d03f"
        dataset = "ae"
        file_name = f"{dataset}_for_unit_tests.zip"
        url = get_repo_url() + file_name
        save_path = os.path.join(get_data_home_dir(), file_name)
        self._path = os.path.join(get_data_home_dir(), dataset)
        download(
            url=url,
            path=save_path,
            sha1_hash=sha1sum_id,
        )
        protected_zip_extraction(
            save_path,
            sha1_hash=sha1sum_id,
            folder=self._path,
        )
        self._train_df = pd.read_csv(os.path.join(self._path, "train.csv"), index_col=0)
        self._test_df = pd.read_csv(os.path.join(self._path, "test.csv"), index_col=0)
        self._train_df.reset_index(drop=True, inplace=True)
        self._test_df.reset_index(drop=True, inplace=True)

        print(f"train sample num: {len(self._train_df)}")
        print(f"test sample num: {len(self._test_df)}")

    @property
    def path(self):
        return self._path

    @property
    def feature_columns(self):
        return [
            "product_name",
            "brand_name",
            "product_category",
            "retailer",
            "description",
            "rating",
            "review_count",
            "style_attributes",
            "total_sizes",
            "available_size",
            "color",
        ]

    @property
    def label_columns(self):
        return ["price"]

    @property
    def train_df(self):
        return self._train_df

    @property
    def test_df(self):
        return self._test_df

    @property
    def metric(self):
        return "r2"

    @property
    def problem_type(self):
        return REGRESSION


class AmazonReviewSentimentCrossLingualDataset:
    def __init__(
        self,
    ):
        sha1sum_id = "9c701aa6fc42ec3fe429bfe85a8dac4532ab9fcd"
        dataset = "amazon_review_sentiment_cross_lingual"
        file_name = f"{dataset}.zip"
        url = get_repo_url() + file_name
        save_path = os.path.join(get_data_home_dir(), file_name)
        self._path = os.path.join(get_data_home_dir(), dataset)
        download(
            url=url,
            path=save_path,
            sha1_hash=sha1sum_id,
        )
        protected_zip_extraction(
            save_path,
            sha1_hash=sha1sum_id,
            folder=get_data_home_dir(),
        )
        self._train_en_df = pd.read_csv(
            os.path.join(self._path, "en_train.tsv"),
            sep="\t",
            header=None,
            names=["label", "text"],
        ).sample(1000, random_state=123)

        self._test_en_df = pd.read_csv(
            os.path.join(self._path, "en_test.tsv"),
            sep="\t",
            header=None,
            names=["label", "text"],
        ).sample(200, random_state=123)

        self._train_en_df.reset_index(drop=True, inplace=True)
        self._test_en_df.reset_index(drop=True, inplace=True)

        print(f"train sample num: {len(self._train_en_df)}")
        print(f"test sample num: {len(self._test_en_df)}")

    @property
    def path(self):
        return self._path

    @property
    def label_columns(self):
        return ["label"]

    @property
    def train_df(self):
        return self._train_en_df

    @property
    def test_df(self):
        return self._test_en_df


class IDChangeDetectionDataset:
    def __init__(self):
        sha1sum_id = "b4a7f3ad12778d65e2ff9a2e4e7bd002c91a0458"
        dataset = "id_change_detection"
        file_name = f"{dataset}_for_unit_tests.zip"
        url = get_repo_url() + file_name
        save_path = os.path.join(get_data_home_dir(), file_name)
        self._path = os.path.join(get_data_home_dir(), dataset)
        download(
            url=url,
            path=save_path,
            sha1_hash=sha1sum_id,
        )
        protected_zip_extraction(
            save_path,
            sha1_hash=sha1sum_id,
            folder=self._path,
        )
        # Extract
        self._train_df = pd.read_csv(os.path.join(self._path, "train.csv"), index_col=0)

        self._test_df = pd.read_csv(os.path.join(self._path, "test.csv"), index_col=0)

        for img_col in self.image_columns:
            self._train_df[img_col] = self._train_df[img_col].apply(
                lambda ele: path_expander(ele, base_folder=os.path.join(self._path, "images"))
            )
            self._test_df[img_col] = self._test_df[img_col].apply(
                lambda ele: path_expander(ele, base_folder=os.path.join(self._path, "images"))
            )

    @property
    def feature_columns(self):
        return ["Previous Image", "Current Image", "Product Title", "Product Type"]

    @property
    def label_columns(self):
        return ["Is Identity Changed?"]

    @property
    def train_df(self):
        return self._train_df

    @property
    def test_df(self):
        return self._test_df

    @property
    def image_columns(self):
        return ["Previous Image", "Current Image"]

    @property
    def metric(self):
        return "roc_auc"

    @property
    def problem_type(self):
        return BINARY

    @property
    def match_label(self):
        return 0


class Flickr30kDataset:
    def __init__(self):
        sha1sum_id = "9f748e009f51ce4013a4244861813220fa1eb517"
        dataset = "flickr30k"
        file_name = f"{dataset}_for_unit_tests.zip"
        url = get_repo_url() + file_name
        save_path = os.path.join(get_data_home_dir(), file_name)
        self._path = os.path.join(get_data_home_dir(), dataset)
        download(
            url=url,
            path=save_path,
            sha1_hash=sha1sum_id,
        )
        protected_zip_extraction(
            save_path,
            sha1_hash=sha1sum_id,
            folder=self._path,
        )
        # Extract
        self._train_df = pd.read_csv(os.path.join(self._path, "train.csv"), index_col=0)

        self._val_df = pd.read_csv(os.path.join(self._path, "val.csv"), index_col=0)

        self._test_df = pd.read_csv(os.path.join(self._path, "test.csv"), index_col=0)

        for img_col in self.image_columns:
            self._train_df[img_col] = self._train_df[img_col].apply(
                lambda ele: path_expander(ele, base_folder=os.path.join(self._path, "images"))
            )
            self._val_df[img_col] = self._val_df[img_col].apply(
                lambda ele: path_expander(ele, base_folder=os.path.join(self._path, "images"))
            )
            self._test_df[img_col] = self._test_df[img_col].apply(
                lambda ele: path_expander(ele, base_folder=os.path.join(self._path, "images"))
            )

    @property
    def feature_columns(self):
        return ["image", "caption"]

    @property
    def label_columns(self):
        return None

    @property
    def train_df(self):
        return self._train_df

    @property
    def val_df(self):
        return self._val_df

    @property
    def test_df(self):
        return self._test_df

    @property
    def image_columns(self):
        return ["image"]

    @property
    def metric(self):
        return "ndcg"

    @property
    def problem_type(self):
        return None

    @property
    def match_label(self):
        return None
