import pandas as pd
from autogluon.multimodal import MultiModalPredictor

train_data = pd.DataFrame({"text": ["high", "higher", "low"], "rating": [2,3,0]})
target_column = "rating"

# AutoML
predictor = MultiModalPredictor(
    label=target_column
)

predictor.fit(
    train_data,
    hyperparameters={
        'env.num_gpus': 1,
        'model.hf_text.checkpoint_name': 'indobenchmark/indobert-lite-large-p2',
    }
)






