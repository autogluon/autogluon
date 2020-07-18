# Run GLUE Tasks with AutoGluon Text

## Prepare the data
```bash
python prepare_glue.py --benchmark glue
```

Train on SST-2 dataset in the GLUE benchmark

```bash
python run_text_prediction.py \
     --do_train \
     --train_file glue/sst/train.parquet \
     --dev_file glue/sst/dev.parquet \
     --test_file glue/sst/test.parquet \
     --task sst
```
