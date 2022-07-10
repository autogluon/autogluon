import pytest
from autogluon.multimodal import MultiModalPredictor
from unittest_datasets import AmazonReviewSentimentCrossLingualDataset
from autogluon.multimodal.constants import LORA_BIAS, LORA_NORM, NORM_FIT, BIT_FIT


@pytest.mark.parametrize("backbone,gradient_checkpointing,efficient_finetuning,pooling_mode,precision",
                         [('google/mt5-small', True, LORA_NORM, 'mean', 'bf16'),
                          ('microsoft/deberta-v3-small', True, BIT_FIT, 'mean', '16')])
def test_predictor_gradient_checkpointing(backbone, efficient_finetuning, pooling_mode, precision):
    dataset = AmazonReviewSentimentCrossLingualDataset()
    train_data = dataset.train_df.sample(200)
    test_data = dataset.test_df.sample(50)
    predictor = MultiModalPredictor(label=dataset.label_columns[0])
    predictor.fit(train_data,
                  hyperparameters={
                      "model.hf_text.checkpoint_name": backbone,
                      "model.hf_text.pooling_mode": pooling_mode,
                      "model.hf_text.gradient_checkpointing": True,
                      "optimization.efficient_finetune": efficient_finetuning,
                      "optimization.lr_decay": 1.0,
                      "optimization.learning_rate": 1e-03,
                      "env.num_gpus": 1,
                  })
    predictions = predictor.predict(test_data)
