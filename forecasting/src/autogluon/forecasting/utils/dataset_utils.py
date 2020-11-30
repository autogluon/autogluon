from gluonts.dataset.repository.datasets import dataset_recipes


def gluonts_builtin_datasets():
    return list(dataset_recipes.keys())
