
import pandas as pd

from tabular.ml.trainer.abstract_trainer import AbstractTrainer
from tabular.ml.models.lgb_model import LGBModel

from tabular.ml.models.sklearn_model import SKLearnModel
from tabular.ml.tuning.autotune import AutoTune

from sklearn.ensemble import (RandomForestClassifier)


# Trainer handles model training details
class Trainer(AbstractTrainer):
    def __init__(self, path, params):
        super().__init__(path=path)
        self.params = params

    def train_rf_ensemble(self, X_train, X_test, y_train, y_test, model_path):
        model_tst = SKLearnModel(path=model_path, name='RandomForestClassifier', model=RandomForestClassifier(n_estimators=350, max_features=0.8, max_depth=32, min_samples_leaf=5, n_jobs=-1))
        oof_pred_proba, y_pred_proba_ensemble, models_cv = self.train_ensemble(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model_base=model_tst)
        print('training stacker...')

        X_train = X_train.join(oof_pred_proba)
        X_test = X_test.join(y_pred_proba_ensemble)

        return self.train_rf(X_train, X_test, y_train, y_test, model_path)

        # model_oof_preds = get_oof_preds(m)

    def train_rf(self, X_train, X_test, y_train, y_test, model_path):
        model_tst = SKLearnModel(path=model_path, name='RandomForestClassifier', model=RandomForestClassifier(n_estimators=70, max_features=0.8, min_samples_leaf=5, max_depth=32, n_jobs=-1))

        autotune = self.autotune(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_base=model_tst)

        model_tst.fit(X_train=X_train, Y_train=y_train)
        print('score:', model_tst.score(X_test, y_test))
        x = model_tst.model.feature_importances_

        a = pd.Series(data=x, index=X_train.columns)
        b = a.sort_values(ascending=False)

        print(b)

        # cnf_matrix = confusion_matrix(y_test, y_pred)
        # fig = plt.figure()
        # fig.set_size_inches(32, 32, forward=True)
        # classes = [self.label_cleaner.cat_mappings_dependent_var[x] for x in range(len(y_train.cat.categories))]
        # plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False, title='Normalized confusion matrix')

        # TODO: Divide # of splits used by feature gain to get average gain per split
        results = model_tst.debug_feature_gain(X_test=X_test, Y_test=y_test, model=model_tst)
        features_to_prune = list(results[results <= 0].index)
        results = results.to_frame(name='accuracy_diff').join(a.to_frame(name='splits'))
        results['avg_gain'] = results['accuracy_diff'] / results['splits'] * 100
        print(results)

        features_to_prune = list(results[results['avg_gain'] <= -50].index)

        print(features_to_prune)

        # FIXME: TMP LOGIC
        # model_tst.save()
        # save_pointer.save(path=model_tst.path + 'model_checkpoint_latest.pointer', content_path=model_tst.model_file_path)
        return model_tst

    # TODO: Add CV functions
    def train_lgbm(self, X_train, X_test, y_train, y_test, model_path):
        model_trainer = LGBModel(name='tst_model', path=model_path, params=self.params, num_boost_round=self.num_boost_round)
        model_trainer.fit(X_train=X_train, Y_train=y_train, X_test=X_test, Y_test=y_test)

        return model_trainer
