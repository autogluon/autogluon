import gc
import sys
import warnings

import kaggle_feedback_prize_preprocess
import pandas as pd
import torch

from autogluon.multimodal import MultiModalPredictor

warnings.filterwarnings("ignore")
sys.path.append("../input/autogluon-standalone/antlr4-python3-runtime-4.8/antlr4-python3-runtime-4.8/src/")
# !pip install - -no - deps - -no - index - -quiet .. / input / autogluon - standalone / *.whl


data_path = "../input/feedback-prize-effectiveness/"

config_1 = {
    "save_path": "../input/feedback_microsoft-deberta-v3-large/microsoft-deberta-v3-large-cv5-lr-5e-05-mepoch-7",
    "per_gpu_batch_size_evaluation": 2,
    "N_fold": 5,
}
config_2 = {
    "save_path": "../input/roberta-large/roberta-large-cv5-lr-5e-05-mepoch-7",
    "per_gpu_batch_size_evaluation": 2,
    "N_fold": 5,
}


if __name__ == "__main__":
    test_df = kaggle_feedback_prize_preprocess.read_and_process_data(data_path, "test.csv", is_train=False)

    configs = [config_1, config_2]
    weights = [0.6, 0.4]

    all_proba = []
    for config in configs:
        print(config)
        model_proba = []
        for fold in range(config["N_fold"]):
            pretrained_model = MultiModalPredictor.load(path=config["save_path"] + f"_{fold}")
            pretrained_model._config.env.per_gpu_batch_size_evaluation = config["per_gpu_batch_size_evaluation"]
            test_proba = pretrained_model.predict_proba(test_df)
            model_proba.append(test_proba)

            # free up CPU memory
            del pretrained_model
            torch.cuda.empty_cache()
            gc.collect()

        proba_concat = pd.concat(model_proba)
        mean_proba = proba_concat.groupby(level=0).mean()
        all_proba.append(mean_proba)

    result = sum([all_proba[i] * weights[i] for i in range(len(configs))])
    result.to_csv("submission.csv", index=False)
    print(result)
