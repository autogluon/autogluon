import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from autogluon.text.automm.constants import (
    MULTICLASS,
    BINARY,
    REGRESSION,
)
from utils import (
    download,
    get_data_home_dir,
    get_repo_url,
    protected_zip_extraction,
    path_expander,
)


class PetFinderDataset:
    def __init__(self,):
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
        self._train_df = pd.read_csv(os.path.join(self._path, 'train.csv'), index_col=0)
        self._test_df = pd.read_csv(os.path.join(self._path, 'test.csv'), index_col=0)
        for img_col in self.image_columns:
            self._train_df[img_col] = self._train_df[img_col].apply(
                lambda ele: path_expander(ele, base_folder=os.path.join(self._path, "images")))
            self._test_df[img_col] =\
                self._test_df[img_col].apply(
                    lambda ele: path_expander(ele, base_folder=os.path.join(self._path, "images")))
            print(self._train_df[img_col][0])
            print(self._test_df[img_col][0])

        _, self._train_df = train_test_split(
            self._train_df,
            test_size=0.5,
            random_state=np.random.RandomState(123),
            stratify=self._train_df[self.label_columns[0]],
        )
        _, self._test_df = train_test_split(
            self._test_df,
            test_size=0.2,
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
        return ['Type', 'Name', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
                'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
                'VideoAmt', 'Description', 'PhotoAmt',
                'Images']

    @property
    def label_columns(self):
        return ['AdoptionSpeed']

    @property
    def train_df(self):
        return self._train_df

    @property
    def test_df(self):
        return self._test_df

    @property
    def image_columns(self):
        return ['Images']

    @property
    def metric(self):
        return 'quadratic_kappa'

    @property
    def problem_type(self):
        return MULTICLASS


class HatefulMeMesDataset:
    def __init__(self,):
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
        self._train_df = pd.read_csv(os.path.join(self._path, 'train.csv'), index_col=0)
        self._test_df = pd.read_csv(os.path.join(self._path, 'test.csv'), index_col=0)
        for img_col in self.image_columns:
            self._train_df[img_col] = self._train_df[img_col].apply(
                lambda ele: path_expander(ele, base_folder=os.path.join(self._path, "images")))
            self._test_df[img_col] =\
                self._test_df[img_col].apply(
                    lambda ele: path_expander(ele, base_folder=os.path.join(self._path, "images")))
            print(self._train_df[img_col][0])
            print(self._test_df[img_col][0])
        self._train_df.reset_index(drop=True, inplace=True)
        self._test_df.reset_index(drop=True, inplace=True)

        print(f"train sample num: {len(self._train_df)}")
        print(f"test sample num: {len(self._test_df)}")

    @property
    def path(self):
        return self._path

    @property
    def feature_columns(self):
        return ['img', 'text']

    @property
    def label_columns(self):
        return ['label']

    @property
    def train_df(self):
        return self._train_df

    @property
    def test_df(self):
        return self._test_df

    @property
    def image_columns(self):
        return ['img']

    @property
    def metric(self):
        return 'accuracy'

    @property
    def problem_type(self):
        return BINARY


class AEDataset:
    def __init__(self,):
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
        self._train_df = pd.read_csv(os.path.join(self._path, 'train.csv'), index_col=0)
        self._test_df = pd.read_csv(os.path.join(self._path, 'test.csv'), index_col=0)
        self._train_df.reset_index(drop=True, inplace=True)
        self._test_df.reset_index(drop=True, inplace=True)

        print(f"train sample num: {len(self._train_df)}")
        print(f"test sample num: {len(self._test_df)}")

    @property
    def path(self):
        return self._path

    @property
    def feature_columns(self):
        return ['product_name', 'brand_name', 'product_category',
                'retailer', 'description', 'rating', 'review_count',
                'style_attributes', 'total_sizes', 'available_size', 'color']

    @property
    def label_columns(self):
        return ['price']

    @property
    def train_df(self):
        return self._train_df

    @property
    def test_df(self):
        return self._test_df

    @property
    def metric(self):
        return 'r2'

    @property
    def problem_type(self):
        return REGRESSION


class SanFranciscoAirbnbDataset:
    def __init__(self,):
        sha1sum_id = "652a17f1315ec0961336aa140cf983776400c933"
        dataset = "san_francisco_airbnb"
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
        self._train_df = pd.read_csv(os.path.join(self._path, 'train.csv'), index_col=0)
        self._test_df = pd.read_csv(os.path.join(self._path, 'test.csv'), index_col=0)
        for img_col in self.image_columns:
            self._train_df[img_col] = self._train_df[img_col].apply(
                lambda ele: path_expander(ele, base_folder=os.path.join(self._path, "images")))
            self._test_df[img_col] =\
                self._test_df[img_col].apply(
                    lambda ele: path_expander(ele, base_folder=os.path.join(self._path, "images")))
            print(self._train_df[img_col][0])
            print(self._test_df[img_col][0])

        self._train_df.reset_index(drop=True, inplace=True)
        self._test_df.reset_index(drop=True, inplace=True)

        print(f"train sample num: {len(self._train_df)}")
        print(f"test sample num: {len(self._test_df)}")

    @property
    def path(self):
        return self._path

    @property
    def feature_columns(self):
        return ['name', 'description', 'neighborhood_overview',
                'host_since', 'host_about', 'host_response_time', 'host_response_rate',
                'host_acceptance_rate', 'host_is_superhost', 'host_listings_count',
                'host_total_listings_count', 'host_has_profile_pic',
                'host_identity_verified', 'neighbourhood', 'neighbourhood_cleansed',
                'neighbourhood_group_cleansed', 'latitude', 'longitude',
                'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms',
                'beds', 'amenities', 'minimum_nights', 'maximum_nights',
                'minimum_minimum_nights', 'maximum_minimum_nights',
                'minimum_maximum_nights', 'maximum_maximum_nights',
                'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'has_availability',
                'availability_30', 'availability_60', 'availability_90',
                'availability_365', 'number_of_reviews', 'number_of_reviews_ltm',
                'number_of_reviews_l30d', 'first_review', 'last_review',
                'review_scores_rating', 'review_scores_accuracy',
                'review_scores_cleanliness', 'review_scores_checkin',
                'review_scores_communication', 'review_scores_location',
                'review_scores_value', 'instant_bookable',
                'calculated_host_listings_count',
                'calculated_host_listings_count_entire_homes',
                'calculated_host_listings_count_private_rooms',
                'calculated_host_listings_count_shared_rooms', 'reviews_per_month',
                'image']

    @property
    def label_columns(self):
        return ['price']

    @property
    def train_df(self):
        return self._train_df

    @property
    def test_df(self):
        return self._test_df

    @property
    def image_columns(self):
        return ['image']

    @property
    def metric(self):
        return 'rmse'

    @property
    def test_metric(self):
        return 'r2'

    @property
    def problem_type(self):
        return REGRESSION


class AmazonReviewSentimentCrossLingualDataset:
    def __init__(self,):
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
            os.path.join(self._path, 'en_train.tsv'),
            sep='\t',
            header=None,
            names=['label', 'text'],
        ).sample(1000, random_state=123)

        self._test_en_df = pd.read_csv(
            os.path.join(self._path, 'en_test.tsv'),
            sep='\t',
            header=None,
            names=['label', 'text'],
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
        return ['label']

    @property
    def train_df(self):
        return self._train_en_df

    @property
    def test_df(self):
        return self._test_en_df
