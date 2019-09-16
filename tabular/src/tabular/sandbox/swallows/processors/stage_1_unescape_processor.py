import html
import re
import sys
import unicodedata

from ply.cpp import xrange
from six import unichr

from tabular.sandbox.swallows.processors.preprocessor import AbstractPreprocessor
from tabular.sandbox.swallows.processors.utils import perform_replacements


class UnescapeProcessor(AbstractPreprocessor):

    def __init__(self):
        self.name = "UnescapeProcessor"

    def run(self, context, df):
        all_chars = (unichr(i) for i in xrange(sys.maxunicode))
        control_chars = ''.join(c for c in all_chars if unicodedata.category(c) == 'Cc')
        control_char_re = re.compile('[%s]' % re.escape(control_chars))

        df['details'] = df['details'].astype(str)
        df['details'] = df['details'].map(lambda txt: html.unescape(txt))
        df['details'] = df['details'].str.replace(control_char_re, '')
        df['details'] = df['details'].str.replace(u'\u200b', '')
        df['details'] = df['details'].str.replace('<.*?>', '')
        df['short_description'] = df['short_description'].astype(str)
        df['short_description'] = df['short_description'].map(lambda txt: html.unescape(txt))
        df['short_description'] = df['short_description'].str.replace(control_char_re, '')

        replacements = [
            '[◥﹉◤✖]+',
            '--[-]+',
            '==[=]+',
            '__[_]+',
            '[*]{2,}',
            '¶m=',
        ]
        r = '|'.join(replacements)
        return perform_replacements(df, 'details', {f'({r})': ' '})

        return df
