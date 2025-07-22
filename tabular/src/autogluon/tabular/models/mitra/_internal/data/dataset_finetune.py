from typing import Optional

import numpy as np
import torch

from ..._internal.config.config_run import ConfigRun
from ..._internal.data.dataset_split import make_dataset_split
from ..._internal.config.enums import Task


class DatasetFinetune(torch.utils.data.Dataset):
    """
    The main goal of this class is to generate a dataset for fine-tuning.
    The input data are the full (x_support, y_support, x_query, y_query)
    But these arrays are too large to be pushed through the model at once.
    So here we split query the data into chunks if the query data is too large.
    If the support data is too large, we randomly sample from it.
    Furthermore, we transition from numpy to tensors.
    """

    def __init__(
        self, 
        cfg: ConfigRun,
        x_support: np.ndarray, 
        y_support: np.ndarray, 
        x_query: np.ndarray, 
        y_query: Optional[np.ndarray],
        max_samples_support: int,
        max_samples_query: int,
        rng: np.random.RandomState,
    ):
        """
        :param: max_features: number of features the tab pfn model has been trained on
        """

        self.cfg = cfg
        self.rng = rng
        
        self.x_support = x_support
        self.y_support = y_support
        self.x_query = x_query        
        self.y_query = y_query

        if self.y_query is None:
            self.y_query = np.zeros((self.x_query.shape[0],)) - 1

        self.max_samples_support = max_samples_support
        self.max_samples_query = max_samples_query

        self.x_queries = self.split_in_chunks(self.x_query, max_samples_query)
        self.y_queries = self.split_in_chunks(self.y_query, max_samples_query)

        self.n_samples_support = self.x_support.shape[0]

        # We push the whole training data through the model, unless it is too large
        self.support_size = min(self.max_samples_support, self.n_samples_support)


    def __len__(self):
        return len(self.x_queries)

    def __getitem__(self, idx):

        support_indices = self.rng.choice(
            self.n_samples_support, 
            size=self.support_size, 
            replace=False
        )

        x_support = self.x_support[support_indices]
        y_support = self.y_support[support_indices]

        x_support_tensor = torch.as_tensor(x_support)
        y_support_tensor = torch.as_tensor(y_support)
        x_query_tensor = torch.as_tensor(self.x_queries[idx])
        y_query_tensor = torch.as_tensor(self.y_queries[idx])

        return {
            'x_support': x_support_tensor,
            'y_support': y_support_tensor,
            'x_query': x_query_tensor,
            'y_query': y_query_tensor,
        }
    


    def split_in_chunks(self, x: np.ndarray, batch_size: int) -> list[np.ndarray]:
        """
        Splits the data into chunks of size batch_size
        """

        n_chunks = int(np.ceil(x.shape[0] / batch_size))
        x_chunks = []

        for i in range(n_chunks):
            x_chunks.append(x[i * batch_size: (i + 1) * batch_size])

        return x_chunks

def DatasetFinetuneGenerator(
    cfg: ConfigRun,
    x: np.ndarray, 
    y: np.ndarray, 
    task: Task,
    max_samples_support: int,
    max_samples_query: int,
    rng: np.random.RandomState,
):
    """
    The dataset fine-tune generator is a generator that yields a dataset for fine-tuning.
    The idea is to split the training dataset into a support and query set.
    Every single iteration, the generator yields a different support and query set split.
    The dataset made always has exactly one batch.
    """
        
    while True:

        x_support, x_query, y_support, y_query = make_dataset_split(x=x, y=y, task=task, seed=rng)
        n_samples_support = x_support.shape[0]
        n_samples_query = x_query.shape[0]

        support_size = min(max_samples_support, n_samples_support)
        query_size = min(max_samples_query, n_samples_query)

        dataset_finetune = DatasetFinetune(
            cfg=cfg,
            x_support=x_support[:support_size],
            y_support=y_support[:support_size],
            x_query=x_query[:query_size],
            y_query=y_query[:query_size],
            max_samples_support=max_samples_support,
            max_samples_query=max_samples_query,
            rng=rng,
        )

        yield dataset_finetune