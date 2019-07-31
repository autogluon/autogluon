import gluonnlp as nlp

__all__ = ['get_dataset']

_dataset = {'sst_2': nlp.data.SST_2,
            'glue_sst': nlp.data.GlueSST2,
            'glue_mnli': nlp.data.GlueMNLI,
            'glue_mrpc': nlp.data.GlueMRPC
            }


def get_dataset(name, **kwargs):
    """Returns a dataset by name

    Parameters
    ----------
    name : str
        Name of the dataset.

    Returns
    -------
    Dataset
        The dataset.
    """
    name = name.lower()
    if name not in _dataset:
        err_str = '"%s" is not among the following dataset list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_dataset.keys())))
        raise ValueError(err_str)
    dataset = _dataset[name](**kwargs)
    return dataset
