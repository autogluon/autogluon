import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

import jsonschema
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..constants import AUTOMM, END_OFFSET, ENTITY_GROUP, NER_ANNOTATION, PROBABILITY, START_OFFSET
from .utils import process_ner_annotations

logger = logging.getLogger(AUTOMM)


class NerLabelEncoder:
    """
    Label Encoder for the named entity recognition task.
    """

    def __init__(self, config: DictConfig, entity_map: Optional[dict] = None):
        self.entity_map = entity_map
        self.config = config
        model_config = config.model.ner_text
        self.ner_special_tags = OmegaConf.to_object(model_config.special_tags)
        self.prefix = config.model.names[0]
        self.b_prefix = "B-"
        self.i_prefix = "I-"

    def fit(self, y: pd.Series, x: pd.Series):
        """
        Extract the annotations, check the unique entity groups, and build entity to index mappings.
        """
        _, entity_groups = self.extract_ner_annotations(y)
        self.unique_entity_groups = self.ner_special_tags + entity_groups
        self.entity_map = {entity: index for index, entity in enumerate(self.unique_entity_groups)}
        self.config.entity_map = self.entity_map
        self.inverse_entity_map = {index: entity for index, entity in enumerate(self.unique_entity_groups)}
        logger.debug(f"Unique entity groups in the data: {entity_groups}")

    def extract_ner_annotations(self, y: pd.Series):
        """
        Validate the JSON annotations with predefined JSON schema and convert it into list.

        Parameters
        ----------
        y
            The raw json annotations.

        Returns
        -------
        all_annotations
            A list of NER annotations.
        unique_entity_groups
            A list of unique entity groups.
        """

        # Predefined Json schema
        schema = {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    ENTITY_GROUP: {"type": "string"},
                    START_OFFSET: {"type": "integer"},
                    END_OFFSET: {"type": "integer"},
                },
                "required": [ENTITY_GROUP, START_OFFSET, END_OFFSET],
            },
        }
        all_annotations = []
        all_entity_groups = []
        for _, ner_annotations in y.items():
            json_ner_annotations = json.loads(ner_annotations)  # load the json annotations
            try:
                jsonschema.validate(json_ner_annotations, schema)  # verify the json schema
            except jsonschema.ValidationError as e:
                # Raise an error if the provided json annotations do not match the predefined schema
                raise ValueError(f"The provided json annotations are invalid: {e.message}")
            sentence_annotations = []
            for annot in json_ner_annotations:
                entity_group = annot[ENTITY_GROUP]
                if not (
                    re.match(self.b_prefix, entity_group, re.IGNORECASE)
                    or re.match(self.i_prefix, entity_group, re.IGNORECASE)
                ):
                    all_entity_groups.append(self.b_prefix + entity_group)
                    all_entity_groups.append(self.i_prefix + entity_group)
                else:
                    all_entity_groups.append(entity_group)
                sentence_annotations.append(
                    (
                        (annot[START_OFFSET], annot[END_OFFSET]),
                        entity_group,
                    )
                )
            all_annotations.append(sentence_annotations)
        unique_entity_groups = list(set(all_entity_groups))
        return all_annotations, unique_entity_groups

    def transform(self, y: pd.Series):
        """
        Transform raw JSON annotations to list annotations.

        Parameters
        ----------
        y
            The raw json annotations.

        Returns
        -------
        all_annotations
            A list of NER annotations.
        """
        all_annotations, _ = self.extract_ner_annotations(y)
        return all_annotations

    def transform_label_for_metric(self, y: pd.Series, x: pd.Series, tokenizer):
        """
        Transform raw JSON annotations to word level annotations which can be used
        as references of seqeval evaluation.
        For more information: https://huggingface.co/spaces/evaluate-metric/seqeval

        Parameters
        ----------
        y
            The raw json annotations.
        x
            The raw text data
        tokenizer
            The tokenizer to be used for text tokenization.

        Returns
        -------
        transformed_y
            A list of word level anotations.
        """
        all_annotations, _ = self.extract_ner_annotations(y)
        transformed_y = []
        for annotation, text_snippet in zip(all_annotations, x.items()):
            word_label, _, _, _ = process_ner_annotations(
                annotation, text_snippet[-1], self.entity_map, tokenizer, is_eval=True
            )
            word_label_invers = []
            for l in word_label:
                entity_group = self.inverse_entity_map[l]
                word_label_invers.append(entity_group)
            transformed_y.append(word_label_invers)
        return transformed_y

    def inverse_transform(self, y: List):
        """
        Inverse Transform NER model predictions into human readable dictionary annotations
        which have the same format as the original annotations.

        Parameters
        ----------
        y
            NER model predictions.

        Returns
        -------
        pred_label_only
            Predictions which only have labels.
        pred_with_offset
            Predictions with both labels and the position (character offset) of the corresponding words.
        """
        pred_label_only, pred_with_offset, pred_with_proba = [], [], []
        for token_preds, offsets, pred_proba in y:
            temp_pred, temp_offset, temp_proba = [], [], []
            for token_pred, offset, probability in zip(token_preds, offsets, pred_proba):
                inverse_pred_label = self.inverse_entity_map[token_pred]
                temp_pred.append(inverse_pred_label)
                temp_proba.append(
                    {
                        ENTITY_GROUP: inverse_pred_label,
                        START_OFFSET: offset[0],
                        END_OFFSET: offset[1],
                        PROBABILITY: {e: p for e, p in zip(list(self.entity_map.keys())[1:], probability[1:])},
                    }
                )
                if inverse_pred_label not in self.ner_special_tags:
                    temp_offset.append(
                        {
                            ENTITY_GROUP: inverse_pred_label,
                            START_OFFSET: offset[0],
                            END_OFFSET: offset[1],
                        }
                    )
            pred_label_only.append(temp_pred)
            pred_with_offset.append(temp_offset)
            pred_with_proba.append(temp_proba)
        return pred_label_only, pred_with_offset, pred_with_proba
