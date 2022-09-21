import random
from typing import Tuple, Union, List

from nlpaug import Augmenter
from nlpaug.util import Method

from .preprocess_dataframe import MultiModalFeaturePreprocessor
from .collator import Dict


def extract_value_from_config(
    config: dict,
    keys: Tuple[str, ...],
):
    """
    Traverse a config dictionary to get some hyper-parameter's value.

    Parameters
    ----------
    config
        A config dictionary.
    keys
        The possible names of a hyper-parameter.

    Returns
    -------
    The hyper-parameter value.
    """
    result = []
    for k, v in config.items():
        if k in keys:
            result.append(v)
        elif isinstance(v, dict):
            result += extract_value_from_config(v, keys)
        else:
            pass

    return result


class InsertPunctuation(Augmenter):
    """
    Inherit nlpaug basic augmenter to support insert random punction at random location https://arxiv.org/pdf/2108.13230.pdf

    example:
    a healthy ,clean , sweet little girl in Mantin . send me message if you can give her a nice home
    ? a ! healthy ,clean , sweet little : girl , in Mantin . send me message . if you ; can give her ? a nice home
    """

    def __init__(
        self,
        name="Insert_Punc",
        aug_min=1,
        aug_max=50,
        aug_p=0.3,
    ):
        """
        Parameters
        ----------
        name
            name used when print out augmentation function
        aug_min
            minimum number of punctuation to insert
        aug_max
            maximum number of punctuation to insert
        aug_p
            how many punctuation to insert calculated as aug_p * sentence length
        """
        super().__init__(
            name=name,
            method=Method.WORD,
            action="insert",
            aug_min=aug_min,
            aug_max=aug_max,
            aug_p=aug_p,
            device="cpu",
            include_detail=False,
            verbose=0,
        )
        self.punc_list = [".", ",", "!", "?", ";", ":"]

    def insert(self, data):
        """
        Random insert random punctuation at random location https://arxiv.org/pdf/2108.13230.pdf

        Parameters
        --------
        data: text


        Returns
        --------
        The augmented text

        """
        words = data.split(" ")
        cnt = random.randint(1, int(self.aug_p * len(words)))
        loc = random.sample(range(0, len(words)), cnt)
        new = []

        for i, word in enumerate(words):
            if i in loc:
                new.append(self.punc_list[random.randint(0, len(self.punc_list) - 1)])
                new.append(word)
            else:
                new.append(word)

        new = " ".join(new)
        return new

    @classmethod
    def clean(cls, data):
        if isinstance(data, list):
            return [d.strip() if d else d for d in data]
        return data.strip()

    @classmethod
    def is_duplicate(cls, dataset, data):
        for d in dataset:
            if d == data:
                return True
        return False


def get_collate_fn(
    df_preprocessor: Union[MultiModalFeaturePreprocessor, List[MultiModalFeaturePreprocessor]],
    data_processors: Union[dict, List[dict]],
):
    """
    Collect collator functions for each modality input of every model.
    These collator functions are wrapped by the "Dict" collator function,
    which can then be used by the Pytorch DataLoader.

    Returns
    -------
    A "Dict" collator wrapping other collators.
    """
    if isinstance(df_preprocessor, MultiModalFeaturePreprocessor):
        df_preprocessor = [df_preprocessor]
    if isinstance(data_processors, dict):
        data_processors = [data_processors]

    collate_fn = {}
    for per_preprocessor, per_data_processors_group in zip(df_preprocessor, data_processors):
        for per_modality in per_data_processors_group:
            per_modality_column_names = per_preprocessor.get_column_names(modality=per_modality)
            if per_modality_column_names:
                for per_model_processor in per_data_processors_group[per_modality]:
                    collate_fn.update(per_model_processor.collate_fn(per_modality_column_names))
    return Dict(collate_fn)
