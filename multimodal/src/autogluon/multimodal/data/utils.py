from typing import Tuple
from nlpaug import Augmenter
from nlpaug.util import Method
import random


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
            how many punctuation to insert calcualted as aug_p * sentence length
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
