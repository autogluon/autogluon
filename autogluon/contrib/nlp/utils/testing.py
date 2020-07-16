import tempfile
import numpy.testing as npt
import mxnet as mx
import os
from mxnet.util import use_np
from .misc import download
from ..base import get_repo_url


@use_np
def verify_nmt_model(model, batch_size=4,
                     src_seq_length: int = 5,
                     tgt_seq_length: int = 10,
                     atol: float = 1E-5,
                     rtol: float = 1E-5):
    """Verify the correctness of an NMT model. Raise error message if it detects problems.

    Parameters
    ----------
    model :
    batch_size :
    src_seq_length :
    tgt_seq_length :
    atol :
    rtol :

    """
    src_word_sequence = mx.np.random.randint(0, model.src_vocab_size, (batch_size, src_seq_length))
    tgt_word_sequence = mx.np.random.randint(0, model.tgt_vocab_size, (batch_size, tgt_seq_length))
    src_valid_length = mx.np.random.randint(1, src_seq_length, (batch_size,))
    min_tgt_seq_length = max(1, tgt_seq_length - 5)
    tgt_valid_length = mx.np.random.randint(min_tgt_seq_length, tgt_seq_length, (batch_size,))
    full_out = model(src_word_sequence, src_valid_length, tgt_word_sequence, tgt_valid_length)
    if full_out.shape != (batch_size, tgt_seq_length, model.tgt_vocab_size):
        raise AssertionError('The output of NMT model does not match the expected output.'
                             ' Model output shape = {}, Expected (B, T, V) = {}'
                             .format(full_out.shape,
                                     (batch_size, tgt_seq_length, model.tgt_vocab_size)))
    for partial_batch_size in range(1, batch_size + 1):
        for i in range(1, min_tgt_seq_length):
            partial_out = model(src_word_sequence[:partial_batch_size, :],
                                src_valid_length[:partial_batch_size],
                                tgt_word_sequence[:partial_batch_size, :(-i)],
                                tgt_valid_length[:partial_batch_size]
                                - mx.np.array(i, dtype=tgt_valid_length.dtype))
            # Verify that the partial output matches the full output
            for b in range(partial_batch_size):
                partial_vl = tgt_valid_length.asnumpy()[b] - i
                npt.assert_allclose(full_out[b, :partial_vl].asnumpy(),
                                    partial_out[b, :partial_vl].asnumpy(), atol, rtol)


@use_np
def verify_nmt_inference(train_model, inference_model,
                         batch_size=4, src_seq_length: int = 5,
                         tgt_seq_length: int = 10,
                         atol: float = 1E-5,
                         rtol: float = 1E-5):
    """Verify the correctness of an NMT inference model. Raise error message if it detects
    any problems.

    Parameters
    ----------
    train_model :
    inference_model :
    batch_size :
    src_seq_length :
    tgt_seq_length :
    atol
        Absolute tolerance
    rtol :
        Relative tolerance
    """
    src_word_sequences = mx.np.random.randint(0, train_model.src_vocab_size,
                                              (batch_size, src_seq_length))
    tgt_word_sequences = mx.np.random.randint(0, train_model.tgt_vocab_size,
                                              (batch_size, tgt_seq_length))
    src_valid_length = mx.np.random.randint(1, src_seq_length, (batch_size,))
    min_tgt_seq_length = max(1, tgt_seq_length - 5)
    tgt_valid_length = mx.np.random.randint(min_tgt_seq_length, tgt_seq_length, (batch_size,))
    full_out = train_model(src_word_sequences, src_valid_length,
                           tgt_word_sequences, tgt_valid_length)
    for partial_batch_size in range(1, batch_size + 1):
        step_out_l = []
        states = inference_model.init_states(src_word_sequences[:partial_batch_size, :],
                                             src_valid_length[:partial_batch_size])
        for i in range(min_tgt_seq_length):
            step_out, states = inference_model(tgt_word_sequences[:partial_batch_size, i], states)
            step_out_l.append(step_out)
        partial_out = mx.np.stack(step_out_l, axis=1)
        npt.assert_allclose(full_out[:partial_batch_size, :min_tgt_seq_length].asnumpy(),
                            partial_out[:partial_batch_size, :].asnumpy(), atol, rtol)


def autonlp_snli_testdata():
    """Get the test SNLI dataset for AutoNLP.

    Returns
    -------
    test_snli_df
    test_snli_metadata
    """
    import pandas as pd
    with tempfile.TemporaryDirectory() as root:
        test_snli_df_path = download(get_repo_url()
                                     + 'autonlp_test_datasets/snli_test_dataset-0de8d633.pd.pkl',
                                     path=os.path.join(root, 'test.pkl'),
                                     sha1_hash='0de8d63354c33d66f34c1e4cc2b4289a9f8c8a3e')
        test_snli_df = pd.read_pickle(test_snli_df_path)
        test_snli_metadata = {'sentence1_entity_numeric': {'type': 'entity',
                                                           'attrs': {'parent': 'sentence1'}},
                              'sentence1_entity_categorical': {'type': 'entity',
                                                               'attrs': {'parent': 'sentence1'}},
                              'sentence2_entity': {'type': 'entity',
                                                   'attrs': {'parent': 'sentence2'}}}
    return test_snli_df, test_snli_metadata
