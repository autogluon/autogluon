# Solve Text Prediction Problems in AutoGluon 

We solve two types of 

## Run GLUE Tasks with AutoGluon Text

### Prepare the data
```bash
python prepare_glue.py --benchmark glue
```

### Run the benchmark
Run on all datasets
 
```bash
for TASK in cola sst mrpc sts qqp qnli rte wnli
do
    TRAIN_FILE=glue/${TASK}/train.parquet
    DEV_FILE=glue/${TASK}/dev.parquet
    TEST_FILE=glue/${TASK}/test.parquet
    python run_text_prediction.py \
     --do_train \
     --train_file ${TRAIN_FILE} \
     --dev_file ${DEV_FILE} \
     --test_file ${TEST_FILE} \
     --task ${TASK}
done

python run_text_prediction.py \
     --do_train \
     --train_file glue/mnli/train.parquet \
     --dev_file glue/mnli/dev_matched.parquet \
     --test_file glue/mnli/test_matched.parquet \
     --task mnli

python run_text_prediction.py \
     --do_train \
     --train_file glue/mnli/train.parquet \
     --dev_file glue/mnli/dev_mismatched.parquet \
     --test_file glue/mnli/test_mismatched.parquet \
     --task mnli
```


