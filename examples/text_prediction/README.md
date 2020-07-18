# Run GLUE Tasks with AutoGluon Text

## Prepare the data
```bash
python prepare_glue.py --benchmark glue
```


```bash
python run_text_classification.py \
     --do_train \
     --train_file glue/sst/train.pd.pkl \
     --dev_file glue/sst/dev.pd.pkl \
     --test_file glue/sst/test.pd.pkl \
     --task sst \
     --batch_size 32 \
     --num_accumulated 1 \
     --ctx gpu0
```
