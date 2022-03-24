from pandas import DataFrame

from autogluon.core.utils.time import sample_df_for_time_func


def test_sample_df_for_time_func():
    df = DataFrame({'a': [1, 5, 3, 8, 12]})

    df_out_1 = sample_df_for_time_func(df=df, sample_size=1, max_sample_size=10)
    df_out_2 = sample_df_for_time_func(df=df, sample_size=3, max_sample_size=10)
    df_out_3 = sample_df_for_time_func(df=df, sample_size=5, max_sample_size=10)
    df_out_4 = sample_df_for_time_func(df=df, sample_size=7, max_sample_size=10)
    df_out_5 = sample_df_for_time_func(df=df, sample_size=14, max_sample_size=10)
    df_out_6 = sample_df_for_time_func(df=df, sample_size=14, max_sample_size=None)
    df_out_7 = sample_df_for_time_func(df=df, sample_size=6, max_sample_size=4)

    assert (df.head(1) == df_out_1).all().all()
    assert (df.head(3) == df_out_2).all().all()
    assert (df == df_out_3).all().all()
    assert 7 == len(df_out_4)
    assert ['a'] == list(df_out_4.columns)
    assert 10 == len(df_out_5)
    assert ['a'] == list(df_out_5.columns)
    assert 14 == len(df_out_6)
    assert ['a'] == list(df_out_6.columns)
    assert 5 == len(df_out_7)
    assert (df == df_out_7).all().all()
