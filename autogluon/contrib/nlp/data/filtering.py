import re
import regex
import requests
import unicodedata
import os
import warnings
from typing import List, Pattern, Union, Tuple, Optional
from sacremoses.normalize import MosesPunctNormalizer
from ..utils.lazy_imports import try_import_fasttext, try_import_langid
from ..utils.misc import download
from ..base import get_model_zoo_home_dir, get_repo_url

non_printing_char_regex = regex.compile(r'\p{C}')


class MosesNormalizer:
    """Normalizes the input sentence. Currently, we support the combination of the

    Moses Punctuation Normalizer 'normalize-punctuation.perl' and the
     'remove-non-printing-char.perl' in [mosesdecoder](https://github.com/moses-smt/mosesdecoder):

    Also, we will normalize the

    Parameters
    ----------
    lang
        The input language
    remove_non_printable_char
        Whether to remove the non-printable unicode characters in the input
    unicode_norm_form
        The unicode normalization format used. Supported

    """
    def __init__(self, lang: str, remove_non_printable_char: bool = True,
                 unicode_norm_form: Optional[str] = None):
        self._remove_non_printable_char = remove_non_printable_char
        self._moses_normalizer = MosesPunctNormalizer(lang)
        self._unicode_norm_form = unicode_norm_form
        if unicode_norm_form is not None:
            assert unicode_norm_form in ['NFC', 'NFKC', 'NFD', 'NFKD'],\
                'Unsupported unicode normalization format, you may refer to ' \
                'https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize for ' \
                'more details.'
        self.__warmup()

    def __warmup(self):
        self('hello world')

    def __call__(self, sentence: str) -> str:
        if self._unicode_norm_form:
            sentence = unicodedata.normalize(self._unicode_norm_form, sentence)
        sentence = self._moses_normalizer.normalize(sentence)
        if self._remove_non_printable_char:
            return non_printing_char_regex.sub(' ', sentence)
        else:
            return sentence


def _words_match_regex(words: List[str], ignore_case=False, replace_white_space=False) -> Pattern:
    """Obtain the regex that finds whether a given corpus contains any word in the input words

    Parameters
    ----------
    words

    Returns
    -------
    regex

    """
    words = [ele for ele in words if ele]
    if ignore_case:
        flags = re.IGNORECASE
    else:
        flags = 0
    if replace_white_space:
        words = [ele.replace(' ', r'\s+') for ele in words]
    regex = re.compile('[^a-z]({words})[^a-z]|^({words})$|^({words})[^a-z]|[^a-z]({words})$'
                       .format(words='|'.join(words)), flags)
    return regex


class ProfanityFilter:
    """Detect whether the corpus contains possible profanity content.

    We use the word list from
     https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words

    """
    def __init__(self, langs: Optional[Union[str, List, Tuple]] = None):
        def _download(url, retries=5):
            while retries + 1 > 0:
                try:
                    r = requests.get(url, stream=True, verify=True)
                    if r.status_code != 200:
                        raise RuntimeError('Failed downloading url {}'.format(url))
                    return r.text
                except Exception as e:
                    retries -= 1
                    if retries <= 0:
                        raise e
                    print('download failed due to {}, retrying, {} attempt{} left'
                          .format(repr(e), retries, 's' if retries > 1 else ''))
        url_path =\
            'https://raw.githubusercontent.com/LDNOOBW/' \
            'List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/' \
            'b36ce5c34c14cb7872dd4c2a4e55fe526138462d/{lang}'
        available_langs = {'ar', 'cs', 'da', 'de', 'en', 'eo', 'es', 'fa', 'fi', 'fr', 'hi', 'hu',
                           'it', 'ja', 'ko', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tlh', 'tr',
                           'zh'}
        self._suspicious_words = []
        if langs is None:
            filter_langs = available_langs
        elif isinstance(langs, str):
            filter_langs = [langs]
        elif isinstance(langs, (tuple, list)):
            filter_langs = list(langs)
        else:
            raise ValueError('Unsupported input langs={}'.format(langs))
        for lang in filter_langs:
            assert lang in available_langs, \
                'lang={} is not supported. All supported languages={}'.format(lang, available_langs)
            out = _download(url_path.format(lang=lang))
            self._suspicious_words += [word.strip() for word in out.split('\n') if word.strip()]
        self._regex = _words_match_regex(self._suspicious_words)

    def match(self, corpus: str) -> bool:
        """Search whether the input corpus contains the suspicious bad words.

        Parameters
        ----------
        corpus
            Input string

        Returns
        -------
        ret
            Whether the input corpus contains profanity words.
        """
        return self._regex.match(corpus) is not None


class LanguageIdentifier:
    """Detect the language of the input corpus.

    We currently support three pretrained models:

        - model='langid'
            Use the langid implementation from
             https://github.com/saffsd/langid.py
        - model='fasttext'
            Use the fasttext model "lid.176.bin" from
             https://fasttext.cc/docs/en/language-identification.html
        - model='fasttext_compressed'
            Use the compressed fasttext model "lid.176.ftz"
            from  https://fasttext.cc/docs/en/language-identification.html

    References:

        @article{joulin2016bag,
          title={Bag of Tricks for Efficient Text Classification},
          author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
          journal={arXiv preprint arXiv:1607.01759},
          year={2016}
        }

        @article{joulin2016fasttext,
          title={FastText.zip: Compressing text classification models},
          author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
          journal={arXiv preprint arXiv:1612.03651},
          year={2016}
        }

        @inproceedings{lui2012langid,
          title={langid. py: An off-the-shelf language identification tool},
          author={Lui, Marco and Baldwin, Timothy},
          booktitle={Proceedings of the ACL 2012 system demonstrations},
          pages={25--30},
          year={2012},
          organization={Association for Computational Linguistics}
        }

    """
    def __init__(self, algo='fasttext_compressed', model_path=None):
        assert algo in ['langid', 'fasttext', 'fasttext_compressed']
        self._algo = algo
        self._use_fasttext = algo.startswith('fasttext')
        if algo in ['fasttext', 'fasttext_compressed']:
            fasttext = try_import_fasttext()
            if model_path is None:
                if algo == 'fasttext':
                    model_path = download(get_repo_url() + 'models/fasttext_langid/lid.176.bin',
                                          os.path.join(get_model_zoo_home_dir(),
                                                       'fasttext_langid', 'lid.176.bin'),
                                          sha1_hash='e613bda316ecb4f5e1924140eedf81b81c087d9a')
                elif algo == 'fasttext_compressed':
                    model_path = download(get_repo_url() + 'models/fasttext_langid/lid.176.ftz',
                                          os.path.join(get_model_zoo_home_dir(),
                                                       'fasttext_langid', 'lid.176.ftz'),
                                          sha1_hash='86d1b630ba55a5040231eda9fe24a7befdc411f2')
                else:
                    raise NotImplementedError
            self._model_path = model_path
            model = fasttext.load_model(model_path)
            self._model = model
        elif algo == 'langid':
            langid = try_import_langid()
            self._model_str = langid.langid.model
            self._model_path = model_path
            self._model = langid.langid.LanguageIdentifier.from_modelstring(self._model_str)
        else:
            raise NotImplementedError

    def __repr__(self):
        s = '{}(algo={}, model_path={})'.format(self.__class__.__name__,
                                                self._algo,
                                                self._model_path)
        return s

    def __call__(self, corpus: str):
        """

        Parameters
        ----------
        corpus
            Input corpus

        Returns
        -------
        lang_label
            The ISO-639 1 code of the predicted language
        score
            The score of the prediction
        """
        if self._use_fasttext:
            corpus = corpus.replace('\n', '')
            labels, scores = self._model.predict(corpus)
            label = labels[0].replace("__label__", "")
            return label, scores[0]
        else:
            return self._model.classify(corpus.lower())

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items() if k != '_model'}
        return d

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        if self._use_fasttext:
            fasttext = try_import_fasttext()
            self._model = fasttext.load_model(self._model_path)
        else:
            langid = try_import_langid()
            self._model = langid.langid.LanguageIdentifier.from_modelstring(self._model_str)
