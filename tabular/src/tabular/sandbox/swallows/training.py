from IPython.lib.display import FileLink
from fastai.tabular import *
from fastai.text import *
from sklearn.model_selection import StratifiedKFold


def gen_splits(n, df, label_col):
    skf = StratifiedKFold(n_splits=n, random_state=42, shuffle=True)
    indexes = range(len(df))
    return skf.split(indexes, df[label_col])


def remove_targets_with_low_frequency(df, target, min_cat_size):
    df_target_counts = pd.DataFrame(df[target].value_counts()).reset_index().rename(columns={target: 'count', 'index': target})
    large_cats = df_target_counts[df_target_counts['count'] > min_cat_size]
    total_count = int(df_target_counts['count'].sum())
    total_big_enough_covered = float(large_cats['count'].sum() * 100 / df_target_counts['count'].sum())
    print(f'Total cats: {len(df_target_counts)} with {total_count} items')
    print(f'cats w/ >{min_cat_size} items: {len(large_cats)} with {total_big_enough_covered:.2f}% coverage')
    return large_cats


def generate_tabular_data_folds(path, df_train, df_test, df_holdout, splits_idxs, dep_var, cont_names, cat_names, procs):
    if df_holdout is not None:
        holdout = TabularList.from_df(df_holdout[cat_names + cont_names], path=path, cat_names=cat_names, cont_names=cont_names)
    for i, split in enumerate(splits_idxs):
        print(f'Generating tabular fold {i}')
        train_idx, val_idx = split
        test = TabularList.from_df(df_test[cat_names + cont_names], path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
        data = (TabularList.from_df(df_train, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                .split_by_idxs(train_idx, val_idx)
                .label_from_df(cols=dep_var)
                .add_test(test)
                .databunch())
        data.save(f'data_tab_fold_{i}')

        if df_holdout is not None:
            data.add_test(holdout)
            data.save(f'data_tab_fold_{i}_holdout')


def generate_text_data_folds(path, df_train, df_test, df_holdout, splits_idxs, dep_var, text_field, lm_data_name):
    data_lm = load_data(path, f'data-lm-{lm_data_name}.pkl', bs=64)
    if df_test is not None:
        test = TextList.from_df(df_test[[text_field]], path, vocab=data_lm.vocab)
    if df_holdout is not None:
        holdout = TextList.from_df(df_holdout[[text_field]], path, vocab=data_lm.vocab)
    for i, split in enumerate(splits_idxs):
        print(f'Generating text data for {text_field} fold {i}')
        train_idx, val_idx = split
        data = (TextList.from_df(df_train[[text_field, dep_var]], cols=text_field, path=path, vocab=data_lm.vocab)
                .split_by_idxs(train_idx, val_idx)
                .label_from_df(cols=dep_var)
                .databunch())

        if df_test is not None:
            data.add_test(test)

        data.save(f'data_{text_field}_fold_{i}')

        if df_holdout is not None:
            data.add_test(holdout)
            data.save(f'data_{text_field}_fold_{i}_holdout')


def train_folds(path, folds, model_name_prefix, fold_train_func, bs, data_suffix='', backwards=False):
    print('--------------------------------------------------------------------')
    print('!!!!! MAKE SURE THE MODEL IS SAVED IN fold_train_func FUNCTION !!!!!')
    print('--------------------------------------------------------------------')
    for fold in range(folds):
        print(f'---------------------------------------------------------------------------------------------------- fold {fold + 1} / {folds}')
        model_name = f'{model_name_prefix}_fold_{fold}'
        output_model_name = f'{model_name}_fitted'
        if backwards:
            output_model_name = output_model_name + '_bwd'
        print(f'Training {model_name} -> {output_model_name}')

        data_file_name = f'{model_name}{data_suffix}'
        data = load_data(path, data_file_name, bs=bs, backwards=backwards)
        print(f'Loaded {data_file_name} | train: {len(data.train_ds)} | valid: {len(data.valid_ds)} | test {len(data.test_ds) if data.test_ds is not None else "NA"} | backwards {backwards}')

        learn = fold_train_func(data, output_model_name)

        # Cleanup
        del learn
        gc.collect()
        torch.cuda.empty_cache()


def predict_on_folds(path, folds, model_name_prefix, learner_create_func, bs, nlp_ordered=False,
                     predict_on_holdout=False, predict_on_test=True, predict_on_valid=True, backwards=False):
    for fold in range(folds):
        print(f'---------------------------------------------------------------------------------------------------- fold {fold + 1} / {folds}')
        model_name = f'{model_name_prefix}_fold_{fold}'
        fitted_model_name = f'{model_name}_fitted'
        if backwards:
            fitted_model_name = fitted_model_name + '_bwd'

        # Test+oof predictions
        data = load_model_data(path, model_name, '', bs, backwards)

        columns = [f'{i}' for i in range(len(data.classes))]

        learn = learner_create_func(data)
        learn.load(fitted_model_name)

        out_name = model_name + ('_bwd' if backwards else '')
        if predict_on_test:
            out_path = path / f'preds/test_{out_name}.parquet'
            preds, y = learn.get_preds(ds_type=DatasetType.Test, ordered=True) if nlp_ordered else learn.get_preds(ds_type=DatasetType.Test)
            pd.DataFrame(preds.numpy(), columns=columns).to_parquet(out_path, engine='fastparquet')
            print(f'Recorded test predictions to {out_path}')

        if predict_on_valid:
            preds, y = learn.get_preds(ds_type=DatasetType.Valid, ordered=True) if nlp_ordered else learn.get_preds(ds_type=DatasetType.Valid)
            out_path = path / f'preds/oof_{out_name}.parquet'
            pd.DataFrame(preds.numpy(), columns=columns).to_parquet(out_path, engine='fastparquet')
            print(f'Recorded OOF predictions to {out_path}')

        # Cleanup
        del learn
        gc.collect()
        torch.cuda.empty_cache()

        # Holdout set prediction
        if predict_on_holdout:
            data = load_model_data(path, model_name, '_holdout', bs, backwards)

            learn = learner_create_func(data)
            learn.load(fitted_model_name)

            preds, y = learn.get_preds(ds_type=DatasetType.Test, ordered=True) if nlp_ordered else learn.get_preds(ds_type=DatasetType.Test)
            out_path = path / f'preds/hold_{out_name}.parquet'
            pd.DataFrame(preds.numpy(), columns=columns).to_parquet(out_path, engine='fastparquet')
            print(f'Recorded holdout predictions to {out_path}')

            # Cleanup
            del learn
            gc.collect()
            torch.cuda.empty_cache()


def bag_folds(path, df_test, model_name_prefix, classes, folds_num, backwards=False):
    itoc = {i: cls for i, cls in enumerate(classes)}

    backwards_suffix = ('_bwd' if backwards else '')

    pred_folds = [pd.read_parquet(path / f'preds/test_{model_name_prefix}_fold_{fold}{backwards_suffix}.parquet', engine='fastparquet') for fold in range(folds_num)]
    mean_preds = np.stack([fold.values for fold in pred_folds]).mean(axis=0)

    columns = [f'{i}' for i in range(len(classes))]
    pd.DataFrame(mean_preds, columns=columns).to_parquet(path / f'preds/test_{model_name_prefix}{backwards_suffix}_bag_mean.parquet', engine='fastparquet')

    df_out = pd.DataFrame({'ID': df_test['ID'], 'root_cause': pd.Series(np.argmax(mean_preds, axis=1)).map(itoc)})
    out_file = f'sub_{model_name_prefix}{backwards_suffix}_bagged.csv'
    df_out.to_csv(out_file, index=False)
    return FileLink(out_file)


def load_model_data(path, model_name, data_suffix, bs, backwards=False):
    data_file_name = f'{model_name}{data_suffix}'
    data = load_data(path, data_file_name, bs=bs, backwards=backwards)
    print(f'Loaded {data_file_name} | train: {len(data.train_ds)} | valid: {len(data.valid_ds)} | test {len(data.test_ds) if data.test_ds is not None else "NA"} | backwards {backwards}')
    return data


def rebuild_oof_predictions_from_folds(path, model_name, splits, backwards=False):
    df = None
    index = np.array([])
    backwards_suffix = '_bwd' if backwards else ''
    splits_idxs = pickle.load(open(path / f'cv_splits-{splits}.pkl', "rb"))
    for fold, split in enumerate(splits_idxs):
        index = np.append(index, split[1])
        input_name = f'preds/oof_{model_name}_fold_{fold}{backwards_suffix}.parquet'
        fold_preds = pd.read_parquet(path / input_name, engine='fastparquet')
        print(f'Loaded {input_name}')
        df = fold_preds if df is None else pd.concat([df, fold_preds])
    df.index = index.astype(int)
    df = df.sort_index()
    output_name = f'preds/oof_{model_name}{backwards_suffix}_all_folds.parquet'
    df.to_parquet(path / output_name, engine='fastparquet')
    print(f'Saved full OOF combined to {output_name}')
    return df


def get_oof_predictions_accuracy(path, model_name, backwards=False):
    classes = load_data(path, f'{model_name}_fold_0').classes
    backwards_suffix = '_bwd' if backwards else ''
    itoc = {i: cls for i, cls in enumerate(classes)}
    oof_preds = pd.read_parquet(path / f'preds/oof_{model_name}{backwards_suffix}_all_folds.parquet', engine='fastparquet')
    data_train = pd.read_parquet(path / 'processed/train.parquet', engine='fastparquet')
    val_preds = pd.Series(np.argmax(oof_preds.values, axis=1)).map(itoc)
    df = pd.DataFrame({'true_label': data_train['root_cause'], 'pred_label': val_preds})
    return np.sum(np.where(df['true_label'] == df['pred_label'], 1, 0)) / len(df)
