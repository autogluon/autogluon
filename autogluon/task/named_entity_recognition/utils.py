"""Some methods are modified from below link
https://github.com/dmlc/gluon-nlp/blob/master/scripts/bert/data/ner.py"""

import re
import sys
from collections import namedtuple
import logging
import mxnet as mx

from seqeval.metrics.sequence_labeling import get_entities

LOG = logging.getLogger(__name__)

TAGGED_TOKEN = namedtuple('TaggedToken', ['text', 'tag'])
PREDICTED_TOKEN = namedtuple('PredictedToken', ['text', 'true_tag', 'pred_tag'])
NULL_TAG = "X"


def read_data(file_path, column_format):
    """Reads the data from the file and convert into list of sentences.

    Parameters
    ----------
    file_path : str
        A file path
    column_format : dict
        A dictionary contains key as an index and value as text/tag to find
        location in a file

    Returns
    -------
    List[List[TaggedToken]]:
        A list of sentences with each sentence breaks down into TaggedToken
    """
    # get the text and ner column
    text_column: int = sys.maxsize
    ner_column: int = sys.maxsize
    for column in column_format:
        if column_format[column].lower() == "text":
            text_column = column
        elif column_format[column].lower() == 'ner':
            ner_column = column
        else:
            raise ValueError("Invalid column type")

    with open(file_path, 'r') as ifp:
        sentence_list = []
        current_sentence = []

        for line in ifp:
            if '-DOCSTART-' in line:
                continue
            if line.startswith("#"):
                continue
            if len(line.strip()) > 0:
                fields = re.split(r'\s+', line.rstrip())
                if len(fields) > ner_column:
                    current_sentence.append(TAGGED_TOKEN(text=fields[text_column],
                                                         tag=fields[ner_column]))
            else:
                # the sentence was completed if an empty line occurred; flush the current sentence.
                if len(current_sentence) > 0:
                    sentence_list.append(current_sentence)
                    current_sentence = []

        # check if there is a remaining token. in most CoNLL data files, this does not happen.
        if len(current_sentence) > 0:
            sentence_list.append(current_sentence)
        return sentence_list


def bio_to_bio2(sentences):
    """Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.

    Parameters
    ----------
    sentences : List[List[TaggedToken]]
        A list of sentences in BIO format with each sentence breaks down into TaggedToken

    Returns
    -------
    List[List[TaggedToken]]:
        A list of sentences in BIO-2 format with each sentence breaks down into TaggedToken
    """
    sentence_list = []
    current_sentence = []
    prev_tag = 'O'

    for sentence in sentences:
        for token in sentence:
            tag = token.tag
            if tag == 'O':
                bio2_tag = 'O'
            else:
                if prev_tag == 'O' or tag[2:] != prev_tag[2:]:
                    bio2_tag = 'B' + tag[1:]
                else:
                    bio2_tag = tag
            current_sentence.append(TAGGED_TOKEN(text=token.text, tag=bio2_tag))
            prev_tag = tag
        sentence_list.append(current_sentence)
        current_sentence = []
        prev_tag = 'O'

    # check if there is a remaining token. in most CoNLL data files, this does not happen.
    if len(current_sentence) > 0:
        sentence_list.append(current_sentence)
    return sentence_list


def bio2_to_bioes(tokens):
    """Convert a list of TaggedTokens from BIO-2 scheme to BIOES scheme.

    Parameters
    ----------
    tokens: List[TaggedToken]
        A list of tokens in BIO-2 scheme

    Returns
    -------
    List[TaggedToken]:
        A list of tokens in BIOES scheme
    """
    ret = []
    for index, token in enumerate(tokens):
        if token.tag == 'O':
            ret.append(token)
        elif token.tag.startswith('B'):
            # if a B-tag is continued by other tokens with the same entity,
            # then it is still a B-tag
            if index + 1 < len(tokens) and tokens[index + 1].tag.startswith("I"):
                ret.append(token)
            else:
                ret.append(TAGGED_TOKEN(text=token.text, tag="S" + token.tag[1:]))
        elif token.tag.startswith('I'):
            # if an I-tag is continued by other tokens with the same entity,
            # then it is still an I-tag
            if index + 1 < len(tokens) and tokens[index + 1].tag.startswith("I"):
                ret.append(token)
            else:
                ret.append(TAGGED_TOKEN(text=token.text, tag="E" + token.tag[1:]))
    return ret


def bert_tokenize_sentence(sentence, bert_tokenizer):
    """Apply BERT tokenizer on a tagged sentence to break words into sub-words.
    This function assumes input tags are following IOBES, and outputs IOBES tags.

    Parameters
    ----------
    sentence: List[TaggedToken]
        List of tagged words
    bert_tokenizer: nlp.data.BertTokenizer
        BERT tokenizer

    Returns
    -------
    List[TaggedToken]: list of annotated sub-word tokens
    """
    ret = []
    for token in sentence:
        # break a word into sub-word tokens
        sub_token_texts = bert_tokenizer(token.text)
        # only the first token of a word is going to be tagged
        ret.append(TAGGED_TOKEN(text=sub_token_texts[0], tag=token.tag))
        ret += [TAGGED_TOKEN(text=sub_token_text, tag=NULL_TAG)
                for sub_token_text in sub_token_texts[1:]]

    return ret


def load_segment(file_path, tokenizer, indexes_format):
    """Load CoNLL format NER datafile with BIO-scheme tags.
    Tagging scheme is converted into BIOES, and words are tokenized into wordpieces
    using `bert_tokenizer`.

    Parameters
    ----------
    file_path: str
        Path of the file
    tokenizer: nlp.data.BERTTokenizer
    indexes_format: dict
        column format of dataset

    Returns
    -------
    List[List[TaggedToken]]: List of sentences, each of which is the list of `TaggedToken`s.
    """
    sentences = read_data(file_path, indexes_format)
    bio2_sentences = bio_to_bio2(sentences)
    bioes_sentences = [bio2_to_bioes(sentence) for sentence in bio2_sentences]
    subword_sentences = [bert_tokenize_sentence(sentence, tokenizer) for sentence in
                         bioes_sentences]

    LOG.info('load %s, its max seq len: %d',
             file_path, max(len(sentence) for sentence in subword_sentences))

    return subword_sentences


def convert_arrays_to_text(text_vocab, tag_vocab, np_text_ids, np_true_tags, np_pred_tags,
                           np_valid_length):
    """Convert numpy array data into text
    Parameters
    ----------
    np_text_ids: token text ids (batch_size, seq_len)
    np_true_tags: tag_ids (batch_size, seq_len)
    np_pred_tags: tag_ids (batch_size, seq_len)
    np.array: valid_length (batch_size,) the number of tokens until [SEP] token
    Returns
    -------
    List[List[PredictedToken]]:
    """
    predictions = []
    for sample_index in range(np_valid_length.shape[0]):
        sample_len = np_valid_length[sample_index]
        entries = []
        for i in range(1, sample_len - 1):
            token_text = text_vocab.idx_to_token[np_text_ids[sample_index, i]]
            true_tag = tag_vocab.idx_to_token[int(np_true_tags[sample_index, i])]
            pred_tag = tag_vocab.idx_to_token[int(np_pred_tags[sample_index, i])]
            # we don't need to predict on NULL tags
            if true_tag == NULL_TAG:
                last_entry = entries[-1]
                entries[-1] = PREDICTED_TOKEN(text=last_entry.text + token_text,
                                              true_tag=last_entry.true_tag,
                                              pred_tag=last_entry.pred_tag)
            else:
                entries.append(
                    PREDICTED_TOKEN(text=token_text, true_tag=true_tag, pred_tag=pred_tag))

        predictions.append(entries)
    return predictions


class F1Ner(mx.metric.EvalMetric):
    """Evaluation F-1 score metric for NER"""
    def __init__(self):
        super().__init__(name='f1_ner')
        self.value = float('nan')

    def update(self, labels, preds):
        true_entities = set(get_entities(labels))
        pred_entities = set(get_entities(preds))

        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        self.value = 2 * p * r / (p + r) if p + r > 0 else 0

    def get(self):
        return (self.name, self.value)

    def reset(self):
        self.value = float('nan')


class AccNer(mx.metric.EvalMetric):
    """Evaluation Accuracy metric for NER"""
    def __init__(self):
        super().__init__(name='acc_ner')
        self.value = float('nan')

    def update(self, labels, preds, flag_nonnull_tag):
        if not isinstance(labels, list):
            labels = [labels]
            preds = [preds]
            flag_nonnull_tag = [flag_nonnull_tag]
        self.value = 0
        for p, l, f in zip(preds, labels, flag_nonnull_tag):
            pred_tags = p.argmax(axis=-1)
            num_tag_preds = f.sum().asscalar()
            self.value += ((pred_tags == l) * f).sum().asscalar() / num_tag_preds
        self.value /= len(labels)


    def get(self):
        return (self.name, self.value)

    def reset(self):
        self.value = float('nan')
