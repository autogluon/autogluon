from collections import namedtuple


# NER utils

TaggedToken = namedtuple('TaggedToken', ['text', 'tag'])
PredictedToken = namedtuple('PredictedToken', ['text', 'true_tag', 'pred_tag'])

NULL_TAG = "X"

def bio_bioes(tokens):
    """Convert a list of TaggedTokens in BIO(2) scheme to BIOES scheme.

    Parameters
    ----------
    tokens: List[TaggedToken]
        A list of tokens in BIO(2) scheme

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

def read_bio_as_bio2(data_path):
    """Read CoNLL-formatted text file in BIO scheme in given path as sentences in BIO2 scheme.

    Parameters
    ----------
    data_path: str
        Path of the data file to read

    Returns
    -------
    List[List[TaggedToken]]:
        List of sentences, each of which is a List of TaggedTokens
    """

    with open(data_path, 'r') as ifp:
        sentence_list = []
        current_sentence = []
        prev_tag = 'O'

        for line in ifp:
            if len(line.strip()) > 0:
                word, _, _, tag = line.rstrip().split(" ")
                # convert BIO tag to BIO2 tag
                if tag == 'O':
                    bio2_tag = 'O'
                else:
                    if prev_tag == 'O' or tag[2:] != prev_tag[2:]:
                        bio2_tag = 'B' + tag[1:]
                    else:
                        bio2_tag = tag
                current_sentence.append(TaggedToken(text=word, tag=bio2_tag))
                prev_tag = tag
            else:
                # the sentence was completed if an empty line occurred; flush the current sentence.
                sentence_list.append(current_sentence)
                current_sentence = []
                prev_tag = 'O'

        # check if there is a remaining token. in most CoNLL data files, this does not happen.
        if len(current_sentence) > 0:
            sentence_list.append(current_sentence)
        return sentence_list


def remove_docstart_sentence(sentences):
    """Remove -DOCSTART- sentences in the list of sentences.

    Parameters
    ----------
    sentences: List[List[TaggedToken]]
        List of sentences, each of which is a List of TaggedTokens; this list may contain DOCSTART sentences.

    Returns
    -------
        List of sentences, each of which is a List of TaggedTokens; this list does not contain DOCSTART sentences.
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
    ret = []
    for token in sentence:
        # break a word into sub-word tokens
        sub_token_texts = bert_tokenizer(token.text)
        # only the first token of a word is going to be tagged
        ret.append(TaggedToken(text=sub_token_texts[0], tag=token.tag))
        ret += [TaggedToken(text=sub_token_text, tag=NULL_TAG)
                for sub_token_text in sub_token_texts[1:]]
    return ret

def load_segment(file_path, tokenizer):
    bio2_sentences = remove_docstart_sentence(read_bio_as_bio2(file_path))
    bioes_sentences = [bio_bioes(sentence) for sentence in bio2_sentences]
    subword_sentences = [bert_tokenize_sentence(sentence, tokenizer) for sentence in bioes_sentences]
    return subword_sentences