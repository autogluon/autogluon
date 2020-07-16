import numpy as np
from typing import List


def get_trimmed_lengths(lengths: List[int],
                        max_length: int,
                        do_merge: bool = False) -> np.ndarray:
    """Get the trimmed lengths of multiple text data. It will make sure that
    the trimmed length is smaller than or equal to the max_length

    - do_merge is True
        Make sure that sum(trimmed_lengths) <= max_length.
        The strategy is to always try to trim the longer lengths.
    - do_merge is False
        Make sure that all(trimmed_lengths <= max_length)

    Parameters
    ----------
    lengths
        The original lengths of each sample
    max_length
        When do_merge is True,
            We set the max_length constraint on the total length.
        When do_merge is False,
            We set the max_length constraint on individual sentences.
    do_merge
        Whether these sentences will be merged

    Returns
    -------
    trimmed_lengths
        The trimmed lengths of the
    """
    lengths = np.array(lengths)
    if do_merge:
        total_length = sum(lengths)
        if total_length <= max_length:
            return lengths
        trimmed_lengths = np.zeros_like(lengths)
        while sum(trimmed_lengths) != max_length:
            remainder = max_length - sum(trimmed_lengths)
            budgets = lengths - trimmed_lengths
            nonzero_idx = (budgets > 0).nonzero()[0]
            nonzero_budgets = budgets[nonzero_idx]
            if remainder // len(nonzero_idx) == 0:
                for i in range(remainder):
                    trimmed_lengths[nonzero_idx[i]] += 1
            else:
                increment = min(min(nonzero_budgets), remainder // len(nonzero_idx))
                trimmed_lengths[nonzero_idx] += increment
        return trimmed_lengths
    else:
        return np.minimum(lengths, max_length)


def match_tokens_with_char_spans(token_offsets: np.ndarray,
                                 spans: np.ndarray) -> np.ndarray:
    """Match the span offsets with the character-level offsets.

    For each span, we perform the following:

    1: Cutoff the boundary

        span[0] = max(span[0], token_offsets[0, 0])
        span[1] = min(span[1], token_offsets[-1, 1])

    2: Find start + end

    We try to select the smallest number of tokens that cover the entity, i.e.,
    we will find start + end, in which tokens[start:end + 1] covers the span.

    We will use the following algorithm:

        For "start", we search for
            token_offsets[start, 0] <= span[0] < token_offsets[start + 1, 0]

        For "end", we search for:
            token_offsets[end - 1, 1] < spans[1] <= token_offsets[end, 1]

    Parameters
    ----------
    token_offsets
        The offsets of the input tokens. Must be sorted.
        That is, it will satisfy
            1. token_offsets[i][0] <= token_offsets[i][1]
            2. token_offsets[i][0] <= token_offsets[i + 1][0]
            3. token_offsets[i][1] <= token_offsets[i + 1][1]
        Shape (#num_tokens, 2)
    spans
        The character-level offsets (begin/end) of the selected spans.
        Shape (#spans, 2)

    Returns
    -------
    token_start_ends
        The token-level starts and ends. The end will also be used.
        Shape (#spans, 2)
    """
    offsets_starts = token_offsets[:, 0]
    offsets_ends = token_offsets[:, 1]
    span_char_starts = spans[:, 0]
    span_char_ends = spans[:, 1]

    # Truncate the span
    span_char_starts = np.maximum(offsets_starts[0], span_char_starts)
    span_char_ends = np.minimum(offsets_ends[-1], span_char_ends)

    # Search for valid start + end
    span_token_starts = np.searchsorted(offsets_starts, span_char_starts, side='right') - 1
    span_token_ends = np.searchsorted(offsets_ends, span_char_ends, side='left')
    return np.concatenate((np.expand_dims(span_token_starts, axis=-1),
                           np.expand_dims(span_token_ends, axis=-1)), axis=-1)


def convert_token_level_span_to_char(token_offsets, token_span_start_ends):
    """Map the token-level start + end of the spans back to the char-level start + ends

    Parameters
    ----------
    token_offsets
        The original token offsets
        Shape (num_token, 2)
    token_span_start_ends
        The token-level starts + ends of span
        Shape (#span, 2)

    Returns
    -------
    char_span_start_ends
        char-level start + end of the span
        Shape (#span, 2)
    """
    char_begins = token_offsets[token_span_start_ends[:, 0], 0]
    char_ends = token_offsets[token_span_start_ends[:, 1], 1]
    return np.concatenate((np.expand_dims(char_begins, axis=-1),
                           np.expand_dims(char_ends, axis=-1)), axis=-1)
