import os
import copy
import logging
from autogluon.core.models.ensemble.stacker_ensemble_model import StackerEnsembleModel
from autogluon.core.models.abstract.abstract_model import AbstractModel

logger = logging.getLogger(__name__)


class LinearAggregatorModel(StackerEnsembleModel):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, **kwargs):
        super()._initialize(**kwargs)
        self.features = list(set(self.features) - set(self.stack_columns))
        if self.model_base is not None:
            self.model_base.base_pred_features = self.feature_metadata.type_group_map_special['stack']
            self.model_base.features = self.features

    # hyperparameter tuning with a separate validation set
    def _hyperparameter_tune(self, X, y, X_val, y_val, scheduler_options, compute_base_preds=True, **kwargs):
        if len(self.models) != 0:
            raise ValueError('self.models must be empty to call hyperparameter_tune, value: %s' % self.models)
        self._add_stack_to_feature_metadata()

        preprocess_kwargs = {'compute_base_preds': compute_base_preds}
        self.model_base.feature_metadata = self.feature_metadata  # TODO: Move this
        self.model_base.set_contexts(self.path + 'hpo' + os.path.sep)
        X = self.preprocess(X=X, preprocess=False, fit=True, **preprocess_kwargs)
        X_val = self.preprocess(X=X_val, preprocess=False, fit=True, **preprocess_kwargs)

        orig_time = scheduler_options[1]['time_out']
        if orig_time:
            scheduler_options[1]['time_out'] = orig_time * 0.8  # TODO: Scheduler doesn't early stop on final model, this is a safety net. Scheduler should be updated to early stop
        hpo_models, hpo_model_performances, hpo_results = self.model_base.hyperparameter_tune(X=X, y=y, X_val=X_val, y_val=y_val, scheduler_options=scheduler_options, **kwargs)
        scheduler_options[1]['time_out'] = orig_time

        bags = {}
        bags_performance = {}
        for i, (model_name, model_path) in enumerate(hpo_models.items()):
            child: AbstractModel = self._child_type.load(path=model_path)

            # TODO: Create new Ensemble Here
            bag = copy.deepcopy(self)
            bag.rename(f"{bag.name}{os.path.sep}T{i}")
            bag.set_contexts(self.path_root + bag.name + os.path.sep)

            bag.model_base = None
            child.rename('')
            child.set_contexts(bag.path + child.name + os.path.sep)
            bag.save_model_base(child.convert_to_template())

            child.rename('S1F1')
            child.set_contexts(bag.path + child.name + os.path.sep)
            if not self.params.get('save_bag_folds', True):
                child.model = None
            if bag.low_memory:
                bag.save_child(child, verbose=False)
                bag.models.append(child.name)
            else:
                bag.models.append(child)
            bag.val_score = child.val_score
            bag._add_child_times_to_bag(model=child)

            bag.save()
            bags[bag.name] = bag.path
            bags_performance[bag.name] = bag.val_score

        # TODO: hpo_results likely not correct because no renames
        return bags, bags_performance, hpo_results
