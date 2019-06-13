import re
import sys
from collections import namedtuple
import mxnet as mx
import logging

log = logging.getLogger(__name__)

TaggedToken = namedtuple('TaggedToken', ['text', 'tag'])

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
                fields = re.split("\s+", line.rstrip())
                if len(fields) > ner_column:
                    current_sentence.append(TaggedToken(text=fields[text_column],
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
        for i, token in enumerate(sentence):
            tag = token.tag
            if tag == 'O':
                bio2_tag = 'O'
            else:
                if prev_tag == 'O' or tag[2:] != prev_tag[2:]:
                    bio2_tag = 'B' + tag[1:]
                else:
                    bio2_tag = tag
            current_sentence.append(TaggedToken(text=token.text, tag=bio2_tag))
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
                ret.append(TaggedToken(text=token.text, tag="S" + token.tag[1:]))
        elif token.tag.startswith('I'):
            # if an I-tag is continued by other tokens with the same entity,
            # then it is still an I-tag
            if index + 1 < len(tokens) and tokens[index + 1].tag.startswith("I"):
                ret.append(token)
            else:
                ret.append(TaggedToken(text=token.text, tag="E" + token.tag[1:]))
    return ret


def remove_docstart_sentence(sentences):
    """Remove -DOCSTART- sentences in the list of sentences.

    Parameters
    ----------
    sentences: List[List[TaggedToken]]
        List of sentences, each of which is a List of TaggedTokens;
        this list may contain DOCSTART sentences.

    Returns
    -------
        List of sentences, each of which is a List of TaggedTokens;
        this list does not contain DOCSTART sentences.
    """
    ret = []
    for sentence in sentences:
        current_sentence = []
        for token in sentence:
            if token.text != '-DOCSTART-':
                current_sentence.append(token)
        if len(current_sentence) > 0:
            ret.append(current_sentence)
    return ret


def bert_tokenize_sentence(sentence, bert_tokenizer):
    """Convert the text word into BERTTokenize format.

    Parameters
    ----------
    sentence : List[List[TaggedToken]]
        A list of sentences with each sentence breaks down into TaggedToken
    bert_tokenizer : gluonnlp.data.transforms
        A Bert Tokenize instance

    Returns
    -------
    List[List[TaggedToken]]:
        A list of sentences in BERT scheme
    """
    ret = []
    for token in sentence:
        # break a word into sub-word tokens
        sub_token_texts = bert_tokenizer(token.text)
        # only the first token of a word is going to be tagged
        ret.append(TaggedToken(text=sub_token_texts[0], tag=token.tag))
        ret += [TaggedToken(text=sub_token_text, tag=NULL_TAG)
                for sub_token_text in sub_token_texts[1:]]
    return ret


def load_segment(file_path, tokenizer, indexes_format):
    sentences = read_data(file_path, indexes_format)
    bio2_sentences = remove_docstart_sentence(bio_to_bio2(sentences))
    bioes_sentences = [bio2_to_bioes(sentence) for sentence in bio2_sentences]
    subword_sentences = [bert_tokenize_sentence(sentence, tokenizer) for sentence in bioes_sentences]
    return subword_sentences

def convert_arrays_to_text(text_vocab, tag_vocab, np_text_ids, np_true_tags, np_pred_tags, np_valid_length):
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
    TaggedToken = namedtuple('TaggedToken', ['text', 'tag'])
    PredictedToken = namedtuple('PredictedToken', ['text', 'true_tag', 'pred_tag'])

    NULL_TAG = "X"
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
                entries[-1] = PredictedToken(text=last_entry.text + token_text,
                                             true_tag=last_entry.true_tag, pred_tag=last_entry.pred_tag)
            else:
                entries.append(PredictedToken(text=token_text, true_tag=true_tag, pred_tag=pred_tag))

        predictions.append(entries)
    return predictions

def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end

def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start

def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


class f1_ner(mx.metric.EvalMetric):
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


class acc_ner(mx.metric.EvalMetric):
    def __init__(self):
        super().__init__(name='acc_ner')
        self.value = float('nan')

    def update(self, labels, preds, flag_nonnull_tag):
        pred_tags = preds.argmax(axis=-1)
        num_tag_preds = flag_nonnull_tag.sum().asscalar()
        self.value = ((pred_tags == labels) * flag_nonnull_tag).sum().asscalar() / num_tag_preds

    def get(self):
        return (self.name, self.value)

    def reset(self):
        self.value = float('nan')
