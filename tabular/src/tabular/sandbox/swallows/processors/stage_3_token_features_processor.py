import re

from tabular.sandbox.swallows.processors.constants import *
from tabular.sandbox.swallows.processors.preprocessor import AbstractPreprocessor
from tabular.sandbox.swallows.processors.utils import perform_replacements, perform_extraction


class Stage3TokenFeaturesProcessor(AbstractPreprocessor):

    def __init__(self):
        self.name = "Stage3TokenFeaturesProcessor"

    def run(self, context, df):
        print('Adding tokens')

        df = perform_extraction(df, 'details', {
            'wfss_workflow': 'WFSS Workflow name: (.+) Hello',
        })

        df = perform_replacements(df, 'details', {
            '[\\"]+': ' ',
            '([Kk]iosk( ID)?|KID)[ ]?[:=\\-]?[ ]?\\d{3,}': f'kiosk {KIOSK}',

            '([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}': MAC,
            '(\\d{1,3}[.]){3}\\d{1,3}': IP,

            '(1Z[A-Z0-9]{7,20}|Tracking ID(/ID)?[s]?([:\\- ]*[0-9A-Z]{5,})+)': TRACKING,
            '(\\WTT|ticket)[ :]*\\d{3,}': TT,
            '[a-z\\-0-9]+@amazon[.a-z]+': AMZ_EMAIL,
            '[a-z\\-0-9]+@[.a-z-]+.(com|co.uk|.eu|.de|.fr|.jp|.co.jp)': OTHER_EMAIL,
            '\\W+[a-zA-Z]+@\\W+': ALIAS,
            '\\W+@[a-zA-Z]+\\W+': ALIAS,

            '(sn-|S/N|SN|serial( number)?|SERIAL (NUMBER)?)[ ]?[:]?[ ]?[0-9A-Za-z\\-]{4,}': SERIAL,
            'serial(ID|[ -]?No[.]?|[ _\\-][Nn]umber|[ ]?[#\\-])?s?[:]?[ ]?([\\dA-Z]{3,})': SERIAL,

            'asset([ -]?ID|No[.]?|[ _\\-]number|[ ]?[#\\-])?s?[:]?[ ]?([\\d]{5,})([ ,]+([\\d]{5,}))*': ASSET,
            '(A/N|Asset|Asset-No.)s?[-:]?[ ]?([\\d]{5,})([ ,]+([\\d]{5,}))*': ASSET,

            'RMA[:]?[ ]?(\\d+ )?(\\d{3,})': RMA,
            'EC2 instance i-\\w{3,}': EC2_INSTANCE,
            'IMEI( number[s])?([: -,]*[\\d]{15})+': IMEI,
            'SIM( number[s]?)?([: \\-,]*[\\dA-Z]{19,20})+': SIMNUM,
            'phone( number)?(([: \\-,]|is)*[\\d\\-\\(\\)]{7,})+': PHONE,
            '[\\da-z\\-.]+.amazon.com[^a-z0-9/.]': HOST,

        })

        print('Adding token features')
        for feature in ALL_FEATURES:
            fname = feature[3:].strip()
            df[f'has_{fname}'] = df['details'].str.contains(feature, flags=re.IGNORECASE)
            print(f'\t- {fname} finished')
        return df
