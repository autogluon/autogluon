from autogluon.eda.base import AbstractAnalysis


def test_api():
    df_train = 'df_train'  # pd.DataFrame()
    df_val = 'df_val'  # pd.DataFrame()
    df_test = 'df_test'  # pd.DataFrame()
    target_col = 'y'

    a = AbstractAnalysis(
        datasets={
            AbstractAnalysis.TRAIN: df_train,
            AbstractAnalysis.VAL: df_val,
            AbstractAnalysis.TEST: df_test,
        },
        target=target_col
    )

    # primitives
    a.with_columns(['a', 'b'])  # apply only to a subset of columns
    a.univariate.hist(bins=20)
    a.univariate.summary()

    a.with_columns()  # All columns
    a.univariate.summary()

    print(f'\nAnalysis:\n\t{a}')


    a.fit()  # fit components which needs fitting
    a.render()  # render
