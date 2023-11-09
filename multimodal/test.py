import os
import warnings

import numpy as np
import time

warnings.filterwarnings("ignore")
np.random.seed(123)

from autogluon.core.utils.loaders import load_pd

train_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet")
test_data = load_pd.load("https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet")
train_data = train_data.sample(1000)

print("train data loaded!")


from autogluon.multimodal import MultiModalPredictor

if __name__ == "__main__":
    model_path = f"Multimodal_distributed-{time.time()}"
    predictor = MultiModalPredictor(
        label="label",
        eval_metric="acc",
        path=model_path,
        hyperparameters={
            "optimization.top_k_average_method": "best",
            "env.num_nodes": 1,
            "env.strategy": "deepspeed_stage_3_offload",
        },
    )
    print("predictor created")
    predictor.fit(train_data, time_limit=180, sync_path="s3://tonyhu-autogluon/multimodal_distributed")
