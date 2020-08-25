# FAQ
:label:`sec_faq`

How can I perform inference on a file that won't fit in memory?

The Tabular Dataset API works with pandas Dataframes, which supports chunking data in to sizes that fit in memory.
Here's an example of one such chunk-based inference:

```{.python .input}
train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
predictor = task.fit(train_data=train_data)

r = requests.get('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv', allow_redirects=True)
open('test.csv', 'wb').write(r.content)
reader = pd.read_csv('test.csv', chunksize=32)
for chunk in reader:
    test_data = task.Dataset(df=chunk)
    predictor.evaluate(test_data)
```

Choose a chunk size that is optimal for your systems available memory.
[Further Reading](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-chunking)
