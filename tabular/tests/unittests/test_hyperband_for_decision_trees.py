from autogluon.tabular import TabularPrediction as task

from autogluon.tabular.learner.default_learner import DefaultLearner as Learner
from autogluon.tabular.features.generators.auto_ml_pipeline import AutoMLPipelineFeatureGenerator

from autogluon.tabular.models.catboost.catboost_model import CatBoostModel
from autogluon.tabular.models.lgb.lgb_model import LGBModel
from autogluon.tabular.models.rf.rf_model import RFModel
from autogluon.tabular.models.xt.xt_model import XTModel


from autogluon.core.space import Int
from autogluon.core.metrics import accuracy

from autogluon.core.scheduler.hyperband import HyperbandScheduler
from autogluon.core.task.base.base_task import compile_scheduler_options

from autogluon.tabular.utils import infer_problem_type

def hyperband_hpo(model_name, label_column='class', strategy="bayesopt",
        num_trials=5, epochs=5, timeout=60*60, thread_per_trial=1):

    eval_metric = accuracy

    train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')

    train_data = train_data.head(500)

    train_size = int(round(len(train_data) * 0.8))
    valid_size = len(train_data) - train_size

    X_train = train_data.head(train_size)
    X_val = train_data.tail(valid_size)

    feat_generator = AutoMLPipelineFeatureGenerator()
    learner = Learner(path_context="/tmp", label=label_column, id_columns=[],
                      feature_generator=feat_generator)
    X, y, X_val, y_val, _, holdout_frac, num_bagging_folds = learner.general_data_processing(X=X_train, X_val=X_val, X_unlabeled=None,
                                                                                holdout_frac=0.1, num_bagging_folds=0)

    prob_type = infer_problem_type(y=train_data[label_column])

    sch_cls = HyperbandScheduler
    model_cls = None

    hyper = {}
    if model_name == "cat":
        hyper = {'iterations': Int(lower=2, upper=25, default=10)}
        model_cls = CatboostModel
    elif model_name == "gbm":
        hyper ={'num_boost_round': Int(lower=2, upper=25, default=20)}
        model_cls = LGBModel
    elif model_name == "xt":
        hyper = {'n_estimators': Int(lower=2, upper=25, default=10)}
        model_cls = XTModel
    elif model_name == "rf":
        hyper = {'n_estimators': Int(lower=2, upper=25, default=10)}
        model_cls = RFModel

    model = model_cls(path="tmp", name=f"{model_name}", problem_type=prob_type,
                              eval_metric=eval_metric, feature_metadata=learner.feature_generator.feature_metadata,
                              hyperparameters=hyper, num_classes=learner.label_cleaner.num_classes)

    scheduler_options = None

    scheduler_options = compile_scheduler_options(
        scheduler_options=scheduler_options,
        search_strategy=strategy,
        search_options=None,
        nthreads_per_trial=thread_per_trial,
        ngpus_per_trial=0,
        checkpoint=None,
        num_trials=num_trials,
        time_out=timeout,
        resume=False,
        visualizer=None,
        time_attr='epoch',
        reward_attr='validation_performance',
        dist_ip_addrs=[],
        )

    hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(X_train=X, y_train=y, X_val=X_val,
                                y_val=y_val, scheduler_options=(sch_cls, scheduler_options), epochs=epochs)

    assert(len(hpo_models)==num_trials)
    for name, res in hpo_model_performances.items():
        assert(res>0)


def test_cat():
    hyperband_hpo("cat")

def test_gbm():
    hyperband_hpo("gbm")

def test_xt():
    hyperband_hpo("xt")

def test_rf():
    hyperband_hpo("rf")

