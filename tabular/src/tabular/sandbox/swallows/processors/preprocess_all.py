from tabular.sandbox.swallows.processors.constants import *
from tabular.sandbox.swallows.processors.preprocess_all_stage_1 import PreprocessorStage1
from tabular.sandbox.swallows.processors.preprocess_all_stage_2 import PreprocessorStage2
from tabular.sandbox.swallows.processors.preprocess_all_stage_3 import PreprocessorStage3
from tabular.sandbox.swallows.processors.preprocess_all_stage_4 import PreprocessorStage4
from tabular.sandbox.swallows.processors.preprocess_all_stage_5 import PreprocessorStage5
from tabular.sandbox.swallows.processors.preprocess_all_stage_6 import PreprocessorStage6

if __name__ == '__main__':
    # TODO: move low-count categories filter from notebook?

    # Sample dataset
    PreprocessorStage1.run_dataset(PATH, DATASET_SAMPLE)
    PreprocessorStage2.run_dataset(PATH, DATASET_SAMPLE)
    PreprocessorStage3.run_dataset(PATH, DATASET_SAMPLE)
    PreprocessorStage4.run_dataset(PATH, DATASET_SAMPLE)
    PreprocessorStage5.run_dataset(PATH, DATASET_SAMPLE)  # DO NOT USE STAGE 5 mean-encoded features - model overfits badly
    PreprocessorStage6.run_dataset(PATH, DATASET_SAMPLE)

    # Full dataset
    # PreprocessorStage1.run_dataset(PATH, DATASET_TEST)
    # PreprocessorStage1.run_dataset(PATH, DATASET_TRAIN)
    # PreprocessorStage2.run_dataset(PATH, DATASET_TRAIN)
    # PreprocessorStage2.run_dataset(PATH, DATASET_TEST, domains_pickle_prefix=DATASET_TRAIN)
    # PreprocessorStage3.run_dataset(PATH, DATASET_TRAIN)
    # PreprocessorStage3.run_dataset(PATH, DATASET_TEST)
    # PreprocessorStage4.run_dataset(PATH, DATASET_TRAIN)
    # PreprocessorStage4.run_dataset(PATH, DATASET_TEST)
    # PreprocessorStage5.run_dataset(PATH, DATASET_TRAIN)
    # PreprocessorStage5.run_dataset(PATH, DATASET_TEST, encoder_pickle_prefix=DATASET_TRAIN)
    # PreprocessorStage6.run_dataset(PATH, DATASET_TRAIN)
    # PreprocessorStage6.run_dataset(PATH, DATASET_TEST, encoder_pickle_prefix=DATASET_TRAIN)
