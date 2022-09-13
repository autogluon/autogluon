import codecs
import os
from typing import Tuple

import pandas as pd
from text_unidecode import unidecode


def get_essay(essay_id: str, input_dir: str, is_train: bool = True) -> str:
    parent_path = input_dir + "train" if is_train else input_dir + "test"
    essay_path = os.path.join(parent_path, f"{essay_id}.txt")
    essay_text = open(essay_path, "r").read()
    return essay_text


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def read_and_process_data(path: str, file: str, is_train: bool) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(path, file))
    df["essay_text"] = df["essay_id"].apply(lambda x: get_essay(x, path, is_train=is_train))
    return df


def read_and_process_data_with_norm(path: str, file: str, is_train: bool) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(path, file))
    df["essay_text"] = df["essay_id"].apply(lambda x: get_essay(x, path, is_train=is_train))
    df["discourse_text"] = df["discourse_text"].apply(resolve_encodings_and_normalize)
    df["essay_text"] = df["essay_text"].apply(resolve_encodings_and_normalize)
    return df
