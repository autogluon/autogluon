from tabular.sandbox.swallows.processors.preprocessor import AbstractPreprocessor
from tabular.sandbox.swallows.processors.utils import perform_replacements


class Stage3CommonTextRemovalProcessor3(AbstractPreprocessor):

    def __init__(self):
        self.name = "Stage3CommonTextRemovalProcessor3"

    def run(self, context, df):
        replacements = [
            'Note:.*Live chat support is now available 24 hours a day, 7 days a week. Visit xxithelp.*OpsTechIT Support Screen Share link: https://screenshare.a2z.com/.*',
        ]
        r = '|'.join(replacements)
        return perform_replacements(df, 'details', {f'({r})': ''})
