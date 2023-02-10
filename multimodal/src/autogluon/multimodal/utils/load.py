import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

from ..constants import AUTOMM
from ..data import NerProcessor, TextProcessor

logger = logging.getLogger(__name__)


def load_text_tokenizers(
    text_processors: Union[List[TextProcessor], List[NerProcessor]],
    path: str,
) -> Union[List[TextProcessor], List[NerProcessor]]:
    """
    Load saved text tokenizers. If text/ner processors already have tokenizers,
    then do nothing.

    Parameters
    ----------
    text_processors
        A list of text/ner processors with tokenizers or their relative paths.
    path
        The root path.

    Returns
    -------
    A list of text/ner processors with tokenizers loaded.
    """
    for per_text_processor in text_processors:
        if isinstance(per_text_processor.tokenizer, str):
            per_path = os.path.join(path, per_text_processor.tokenizer)
            per_text_processor.tokenizer = per_text_processor.get_pretrained_tokenizer(
                tokenizer_name=per_text_processor.tokenizer_name,
                checkpoint_name=per_path,
            )
    return text_processors


class CustomUnpickler(pickle.Unpickler):
    """
    This is to make pickle loading df_preprocessor backward compatible.
    A df_preprocessor object saved with old name space `autogluon.text.automm` has errors
    when being loaded under the context of new name `autogluon.multimodal`.
    """

    def find_class(self, module, name):
        renamed_module = module
        if module.startswith("autogluon.text.automm"):
            renamed_module = module.replace("autogluon.text.automm", "autogluon.multimodal")

        return super(CustomUnpickler, self).find_class(renamed_module, name)
