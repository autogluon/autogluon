from ......miscs import warning_filter
with warning_filter():
    from skopt.space import Integer, Categorical, Real

from ...abstract.abstract_spaces import AbstractSpaces


class LGBMSpaces(AbstractSpaces):
    def __init__(self, problem_type, objective_func, num_classes=None):
        super().__init__(problem_type=problem_type, objective_func=objective_func, num_classes=num_classes)

    def get_binary_baseline(self):
        spaces = [
            [
                Integer(16, 96, name='num_leaves'),
                Integer(2, 10, name='min_data_in_leaf'),
                Real(0.25, 0.9, name='feature_fraction'),
                Categorical(
                    [
                        True,
                        False
                    ], name='is_unbalance'),  # TODO: This could be auto-selected based on optimization function, perhaps do this as a first check in multi-round hyperparameter optimization
                # Categorical(
                #     [
                #         'gbdt',
                #         'dart',
                #     ], name='boosting_type'),
            ],
            [
                Real(0.0, 2.0, name='lambda_l1'),
                Real(0.0, 2.0, name='lambda_l2'),
            ]
        ]
        return spaces

    def get_multiclass_baseline(self):
        return self.get_binary_baseline()

    def get_regression_baseline(self):
        spaces = [
            [
                Integer(64, 256, name='num_leaves'),
                Integer(1, 2, name='min_data_in_leaf'),
                Real(0.8, 1, name='feature_fraction'),
                # Categorical(
                #     [
                #         'gbdt',
                #         'dart',
                #     ], name='boosting_type'),
            ],
            [
                Real(0.0, 2.0, name='lambda_l1'),
                Real(0.0, 2.0, name='lambda_l2'),
            ]
        ]
        return spaces
