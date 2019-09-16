from tabular.sandbox.swallows.processors.constants import *
from tabular.sandbox.swallows.processors.preprocessor import AbstractPreprocessor
from tabular.sandbox.swallows.processors.utils import perform_replacements, perform_extraction
import re


class Stage4Features(AbstractPreprocessor):

    def __init__(self):
        self.name = "Stage4Features"

    @staticmethod
    def cut_low_populated_groups(df, col_to_assess, min_freq):
        group_freq = df[col_to_assess].value_counts().reset_index().rename(columns={'index': 'cat'})
        groups_to_keep = list(group_freq[group_freq[col_to_assess] >= min_freq]['cat'])
        df[col_to_assess] = df[col_to_assess].where(df[col_to_assess].isin(groups_to_keep), '')
        print(f'\t - cut low populated groups for {col_to_assess}: {len(group_freq)} -> {len(groups_to_keep)}')

    def run(self, context, df):
        df['short_description'].fillna('', inplace=True)
        df = perform_replacements(df, 'short_description', {
            '(\\W|^)[A-Z0-9]{4,5}\\W': ' ',
            '\\[\\]': '',
            '([Kk]iosk( ID)?|KID)[ ]?[:=\\-]?[ ]?\\d{3,}': KIOSK,
            # '[a-z\\-0-9.]+.amazon.(com|co.uk|.eu|.de|.fr|.jp|.co.jp)': HOST,
            '(sn-|S/N|SN|[Ss]erial( number)?|SERIAL (NUMBER)?)[ ]?[:]?[ ]?[0-9A-Za-z\\-]{4,}': SERIAL,
            '[Ss]erial(ID| No[.]?|[ _\\-][Nn]umber|[ ]?[#\\-])?s?[:]?[ ]?([\\dA-Z]{3,})': SERIAL,
            'asset([ ]?ID| No[.]?|[ _\\-]number|[ ]?[#\\-])?s?[:]?[ ]?([\\dA-Z]{5,})([ ,]+([\\dA-Z]{5,}))*': ASSET,
            '([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}': MAC,
            '(\\d{1,3}[.]){3}\\d{1,3}': IP,
            '^[ ]?[-:] ': ' ',
            '[\\"]+': '',
        })

        df = perform_extraction(df, 'short_description', {
            'host_status': f'{HOST}(.*)',
        })

        df = perform_replacements(df, 'host_status', {
            '[\\[\\]:/.-]+': ' ',
            '\\W+': ' ',
        })
        df['host_status'] = df['host_status'].fillna('').str.strip().str.lower()

        Stage4Features.cut_low_populated_groups(df, 'host_status', 10)

        return df
