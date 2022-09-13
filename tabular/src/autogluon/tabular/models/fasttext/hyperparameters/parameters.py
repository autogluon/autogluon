__all__ = ["get_param_baseline"]


# https://fasttext.cc/docs/en/python-module.html#train_supervised-parameters
def get_param_baseline():
    params = {
        "lr": 0.1,  # learning rate [0.1]
        "dim": 100,  # size of word vectors [100]
        "ws": 5,  # size of the context window [5]
        "epoch": 50,  # number of epochs [5]
        "minCount": 1,  # minimal number of word occurrences [1]
        "minCountLabel": 1,  # minimal number of label occurrences [1]
        "minn": 2,  # min length of char ngram [0]
        "maxn": 6,  # max length of char ngram [0]
        "neg": 5,  # number of negatives sampled [5]
        "wordNgrams": 3,  # max length of word ngram [1]
        "loss": "softmax",  # loss function {ns, hs, softmax, ova} [softmax]
        "bucket": 2000000,  # number of buckets [2000000]
        # thread: number of threads [number of cpus]
        "lrUpdateRate": 100,  # change the rate of updates for the learning rate [100]
        "t": 0.0001,  # sampling threshold [0.0001]
        # label: prefix ['__label__']
        # pretrainedVectors:  pretrained word vectors (.vec file) for supervised learning []
        # "verbose": 2,
        "quantize_model": True,
    }
    return params.copy()
