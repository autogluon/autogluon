import logging
import subprocess
import pandas as pd
import numpy as np

from autogluon.core.constants import BINARY, MULTICLASS
from .pecos_utils import load_json_multi

logger = logging.getLogger(__name__)


class PecosInterface():
    """
    Class for interfacing with the PECOS command line tools. Currently only supports PECOS XRLinear.
    """
    def __init__(
            self,
            problem_type=BINARY,
            model_type="XRLinear",
            max_leaf_size=None,
            nr_splits=None,
            spherical=None,
            kmeans_max_iter=None,
            solver_type=None,
            coefficient_positive=None,
            coefficient_negative=None,
            bias=None,
            negative_sampling=None,
            sparsity_threshold=None):

        self.model_type = model_type
        self.problem_type = problem_type

        # Configure hyperparameters
        self.max_leaf_size = max_leaf_size
        self.nr_splits = nr_splits
        self.spherical = spherical
        self.kmeans_max_iter = kmeans_max_iter
        self.solver_type = solver_type
        self.coefficient_positive = coefficient_positive
        self.coefficient_negative = coefficient_negative
        self.bias = bias
        self.negative_sampling = negative_sampling
        self.sparsity_threshold = sparsity_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series, timeout=None, workdir=None, model_dir=None):

        # Set up directories
        self.workdir = workdir
        self.model_dir = model_dir
        self.data_file_train = self.workdir / 'data.txt'
        self.data_file_test = self.workdir / 'test-data.txt'
        self.label_dict_file = self.workdir / 'labels.txt'

        # Save X to a file
        with self.data_file_train.open(mode='w') as f:
            for row in X:
                f.write(f"{row}\n")

        # Save label dictionary to a file
        self.label_dict = {label: i for i, label in enumerate(y.unique())}
        with self.label_dict_file.open(mode='w') as f:
            for label in self.label_dict:
                f.write(f'{label}\n')

        self.max_label_value = y.nunique()

        # Run the model
        if self.model_type == 'XRLinear':
            cmd = f'''
python3 -m pecos.apps.text2text.train  \
    -i {self.data_file_train} -q {self.label_dict_file} \
    -m {self.model_dir}  --verbose 2'''
            cmd = self.append_training_hyperparameters(cmd)
        else:
            raise NotImplementedError("XRLinear is the only PECOS model currently supported by AutoGluon")
        logger.info(cmd)

        # Validate output
        try:
            subprocess.check_output(cmd, shell=True, timeout=timeout)
        except subprocess.CalledProcessError as e:
            logger.error(e)
            raise
        logger.info('finish training')

    def append_training_hyperparameters(self, cmd):
        """
        Add hyperparameters to the command used to run PECOS, if they are specified
        """
        if self.max_leaf_size is not None:
            cmd += f' --max-leaf-size {self.max_leaf_size}'
        if self.nr_splits is not None:
            cmd += f' --nr-splits {self.nr_splits}'
        if self.max_leaf_size is not None:
            cmd += f' --max-leaf-size {self.max_leaf_size}'
        if self.solver_type is not None:
            cmd += f' --solver-type {self.solver_type}'
        if self.coefficient_positive is not None:
            cmd += f' --Cp {self.coefficient_positive}'
        if self.coefficient_negative is not None:
            cmd += f' --Cn {self.coefficient_negative}'
        if self.bias is not None:
            cmd += f' --bias {self.bias}'
        if self.negative_sampling is not None:
            cmd += f' --negative-sampling {self.negative_sampling}'
        if self.sparsity_threshold is not None:
            cmd += f' --threshold {self.sparsity_threshold}'
        return cmd

    def predict(self,  X: np.ndarray):
        df_pred = self.predict_proba(X, k=1)
        return df_pred

    def predict_proba(self, X: pd.DataFrame, k=1):
        # Save X to a file
        with self.data_file_test.open(mode='w') as f:
            for row in X:
                f.write(f'{row}\n')

        # Run the model
        if self.model_type == 'XRLinear':
            cmd = f'''
python3 -m pecos.apps.text2text.predict  \
    -i {self.data_file_test} \
    -m {self.model_dir} -o {self.workdir / 'pred.json'}'''
        else:
            raise NotImplementedError("XRLinear is the only PECOS model currently supported by AutoGluon")
        logger.info(cmd)

        # Validate output
        try:
            subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            logger.error(e)
            raise

        # Read predictions from file
        pred_file = self.workdir / 'pred.json'
        if self.problem_type in [MULTICLASS, BINARY]:
            probs = np.zeros((len(X), self.max_label_value))
            for i, r in enumerate(load_json_multi(pred_file)):
                for ent in r['data']:
                    label = int(ent[0])
                    score = float(ent[1])
                    probs[i, label] = score
                probs[i] /= sum(probs[i])
        else:
            print(f"Problem type not supported: {self.problem_type}")
        return np.array(probs)
