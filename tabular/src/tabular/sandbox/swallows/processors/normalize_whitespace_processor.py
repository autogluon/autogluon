from tabular.sandbox.swallows.processors.preprocessor import AbstractPreprocessor


class NormalizeWhitespaceProcessor(AbstractPreprocessor):

    def __init__(self, field='details'):
        self.name = "NormalizeWhitespaceProcessor"
        self.field = field

    def run(self, context, df):
        to_replace = [
            '[\n\t\r ]+',
        ]
        [df[self.field].replace(rep, ' ', inplace=True, regex=True) for rep in to_replace]
        return df
