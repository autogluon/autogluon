#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:57:46 2020

@author: yirawan
"""

"""BERT dataset transform."""
import collections
from gluonnlp.data import BERTSentenceTransform

#import gluonnlp as nlp
#from gluonnlp.data import SQuAD
from gluonnlp.data.bert.glue import concat_sequences
from gluonnlp.data.bert.squad import improve_answer_span, \
    tokenize_and_align_positions, get_doc_spans, align_position2doc_spans, \
    check_is_max_context, convert_squad_examples

SquadBERTFeautre = collections.namedtuple('SquadBERTFeautre', [
    'example_id', 'qas_id', 'doc_tokens', 'valid_length', 'tokens',
    'token_to_orig_map', 'token_is_max_context', 'input_ids', 'p_mask',
    'segment_ids', 'start_position', 'end_position', 'is_impossible'
])




__all__ = ['BERTDatasetTransform']



class BERTDatasetTransform:

    """Dataset transformation for BERT-style Question-Answering(i.e SQuAD) format.
    Parameters
    ----------
    example : input sequence.
    cls_token : [cls] mark
    sep_token : [sep] mark
    
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    vocab : Vocab or BERTVocab
        The vocabulary.
    doc_stride : int.
        how long the seq would be stride
    max_query_length : int.
        Limit of question length
    """
    def __init__(self, tokenizer=None, cls_token=None, sep_token=None,
                 vocab=None, max_seq_length=384, doc_stride=128, max_query_length=64,
                 cls_index=0):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.cls_index = cls_index
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.max_seq_length = max_seq_length

    def __call__(self, example):
        """Perform transformation for sequence pairs.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - generate answer's position with start and end postion
        - if the question is impossible to answer, set the position to cls index
        - insert [CLS], [SEP] as necessary
        - record whether the tokens in a docspan have max context
        - get sequence features: tokens, segment_ids, p_masks
        - generated SquadBERTFeautre as return value

        Parameters
        ----------

        Returns
        -------
        list of SquadBERTFeautre object consist of parameters:
        (tokens, segment_ids, p_mask), (start, end), is_max, t2o

        """
        
        self.query_tokenized = [self.cls_token] + self.tokenizer(
            example.question_text)[:self.max_query_length]
        #tokenize paragraph and get start/end position of the answer in tokenized paragraph
        self.tok_start_position, self.tok_end_position, self.all_doc_tokens, _, \
                                                        self.tok_to_orig_index = \
                                tokenize_and_align_positions(example.doc_tokens,
                                                             example.start_position,
                                                             example.end_position,
                                                             self.tokenizer)
        # get doc spans using sliding window
        self.doc_spans, self.doc_spans_indices = get_doc_spans(
            self.all_doc_tokens, self.max_seq_length - len(self.query_tokenized) - 2,\
                            self.doc_stride)


        if not example.is_impossible:
            (tok_start_position, tok_end_position) = improve_answer_span(
                self.all_doc_tokens, self.tok_start_position, self.tok_end_position, 
                self.tokenizer, example.orig_answer_text)
            # get the new start/end position
            positions = [
                align_position2doc_spans([tok_start_position, tok_end_position],
                                         doc_idx,
                                         offset=len(self.query_tokenized) + 1,
                                         default_value=0)
                for doc_idx in self.doc_spans_indices
            ]
        else:
            # if the question is impossible to answer, set the start/end position to cls index
            positions = [[self.cls_index, self.cls_index] for _ in self.doc_spans_indices]
    
        # record whether the tokens in a docspan have max context
        token_is_max_context = [{
            len(self.query_tokenized) + p:
            check_is_max_context(self.doc_spans_indices, i, p + self.doc_spans_indices[i][0])
            for p in range(len(self.doc_span))
        } for (i, self.doc_span) in enumerate(self.doc_spans)]
    
        token_to_orig_map = [{
            len(self.query_tokenized) + p + 1:
            self.tok_to_orig_index[p + self.doc_spans_indices[i][0]]
            for p in range(len(doc_span))
        } for (i, doc_span) in enumerate(self.doc_spans)]
    
        #get sequence features: tokens, segment_ids, p_masks
        seq_features = [
            concat_sequences([self.query_tokenized, self.doc_span], [[self.sep_token]] * 2)
            for doc_span in self.doc_spans
        ]
    
    
        features = [
            SquadBERTFeautre(example_id=example.example_id,
                             qas_id=example.qas_id,
                             doc_tokens=example.doc_tokens,
                             valid_length=len(tokens),
                             tokens=tokens,
                             token_to_orig_map=t2o,
                             token_is_max_context=is_max,
                             input_ids=self.vocab[tokens],
                             p_mask=p_mask,
                             segment_ids=segment_ids,
                             start_position=start,
                             end_position=end,
                             is_impossible=example.is_impossible)
            for (tokens, segment_ids, p_mask), (start, end), is_max, t2o in zip(
                seq_features, positions, token_is_max_context, token_to_orig_map)
        ]
        return features
