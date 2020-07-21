python run_text_prediction.py \
     --do_train \
     --train_file glue/sst/train.parquet \
     --dev_file glue/sst/dev.parquet \
     --test_file glue/sst/test.parquet \
     --task sst


python run_text_prediction.py \
     --do_train \
     --train_file glue/mrpc/train.parquet \
     --dev_file glue/mrpc/dev.parquet \
     --test_file glue/mrpc/test.parquet \

