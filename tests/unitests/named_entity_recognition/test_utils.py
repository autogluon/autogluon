from __future__ import print_function

import pytest
from autogluon.task.named_entity_recognition.utils import *


@pytest.mark.serial
def test_bio_to_bio2():
    sentences = []
    current_sentence = []
    TAGGED_TOKEN = namedtuple('TaggedToken', ['text', 'tag'])
    text_list = [
        ['Tony', 'Stark', 'is', 'playing', 'at', 'Club', 'Hawai'],
        ['Steve', 'said', 'cold', 'Temperature', 'tomorrow', 'morning'],
        ['Seven', 'months', 'to', 'November', '20', ',', '1995']
    ]
    tag_list = [
        ['B-PER', 'I-PER', 'O', 'O', 'O', 'I-LOC', 'I-LOC'],
        ['B-LOC', 'O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'O']
    ]
    expected_tag_list = [
        ['B-PER', 'I-PER', 'O', 'O', 'O', 'B-LOC', 'I-LOC'],
        ['B-LOC', 'O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'O']
    ]
    for sentence, tags in zip(text_list, tag_list):
        for word, tag in zip(sentence, tags):
            current_sentence.append(TAGGED_TOKEN(text=word, tag=tag))
        sentences.append(current_sentence)
        current_sentence = []

    output_sentence = bio_to_bio2(sentences)

    # Check output tag with the expected tag
    for sentence, tags in zip(output_sentence, expected_tag_list):
        for token, tag in zip(sentence, tags):
            assert token.tag == tag


@pytest.mark.serial
def test_bio2_to_bioes():
    expected_sentences = []
    current_sentence = []
    TAGGED_TOKEN = namedtuple('TaggedToken', ['text', 'tag'])
    text_list = [
        ['UES', 'rejects', 'Austrian', 'call', 'to', 'boycott', 'French', 'lamb', '.'],
        ['Chris', 'Anderson'],
        ['Africa', 'imported', '39230', 'goat', 'from', 'London', 'last', 'year']
    ]
    tag_list = [
        ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O'],
        ['B-PER', 'I-PER'],
        ['B-LOC', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O']
    ]
    expected_tag_list = [
        ['S-ORG', 'O', 'S-MISC', 'O', 'O', 'O', 'S-MISC', 'O', 'O'],
        ['B-PER', 'E-PER'],
        ['S-LOC', 'O', 'O', 'O', 'O', 'S-LOC', 'O', 'O']
    ]
    for sentence, tags in zip(text_list, tag_list):
        for word, tag in zip(sentence, tags):
            current_sentence.append(TAGGED_TOKEN(text=word, tag=tag))
        expected_sentences.append(bio2_to_bioes(current_sentence))
        current_sentence = []

    # Check output tag with the expected tag
    for sentence, tags in zip(expected_sentences, expected_tag_list):
        for token, tag in zip(sentence, tags):
            assert token.tag == tag


@pytest.mark.serial
def test_f1ner():
    labels = ['S-ORG', 'O', 'S-MISC', 'O', 'O', 'O', 'S-MISC', 'O', 'O']
    preds = ['S-ORG', 'O', 'S-MISC', 'O', 'O', 'O', 'S-MISC', 'O', 'O']

    # Test same outout as expected with f1 score as 1
    f1 = F1Ner()
    f1.update(labels, preds)
    assert f1.value == 1.0

    # Test the output tag which is different from expected('O' tag doesn't count),
    # results in f1 score of zero
    preds = ['O', 'O', 'S-LOC', 'O', 'O', 'O', 'O', 'O', 'O']
    f1 = F1Ner()
    f1.update(labels, preds)
    assert f1.value == 0
