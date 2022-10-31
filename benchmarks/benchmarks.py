# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
from autogluon.tabular import TabularDataset, TabularPredictor


'''
def setup(): pass # module level setup (prior )
def teardown(): pass # module level setup (prior )
'''


class TabularSuite:

    def setup_cache(self): # Note: run for each benchmark and each repeat

        subsample_size = 500
        train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
        subsampled_train_data = train_data.sample(n=subsample_size, random_state=0)

        cached_predictor = TabularPredictor(label='class', path='ag_resources').fit(subsampled_train_data)

    def setup(self):
        self.subsample_size = 500
        self.train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
        self.subsampled_train_data = self.train_data.sample(n=self.subsample_size, random_state=0)

        self.predictor = TabularPredictor.load('ag_resources') 

        self.test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
        self.y_test = self.test_data['class']
        self.test_data_nolab = self.test_data.drop(columns=['class'])
        self.y_pred = self.predictor.predict(self.test_data_nolab)

    '''
    def teardown(self): pass # delete resources (maybe AG file)
    '''

    def time_fit_tabular(self):
        _ = TabularPredictor(label='class', path='ag_resources').fit(self.subsampled_train_data)

    def time_predict_tabular(self):
        _ = self.predictor.predict(self.test_data_nolab)

    def mem_tabular_predictor(self):
        return self.predictor

    def track_acc(self):
        perf = self.predictor.evaluate_predictions(y_true=self.y_test, y_pred=self.y_pred, auxiliary_metrics=True)
        return perf['accuracy']
