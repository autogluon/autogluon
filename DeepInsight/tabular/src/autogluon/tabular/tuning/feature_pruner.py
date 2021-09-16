# import warnings
# warnings.filterwarnings('ignore')
import copy, logging
import pandas as pd

logger = logging.getLogger(__name__)


# TODO: currently is buggy
class FeaturePruner:
    def __init__(self, model_base, threshold_baseline=0.004, is_fit=False):
        self.model_base = model_base
        self.threshold_baseline = threshold_baseline
        self.is_fit = is_fit

        self.best_score = 0
        self.best_iteration = 0
        self.features_in_iter = []
        self.score_in_iter = []
        self.valid_feature_counts = []
        self.gain_dfs = []
        self.banned_features_in_iter = []
        self.thresholds = []
        self.threshold = self.threshold_baseline
        self.cur_iteration = 0
        self.tuned = False
        self.early_stopping_rounds = 2

    def evaluate(self):
        untuned_score = self.score_in_iter[0]
        logger.debug('untuned_score: %s' % untuned_score)
        logger.debug('best_score: %s' % self.best_score)
        logger.debug('best_iteration: %s' % self.best_iteration)

        data_dict = {
            'score': self.score_in_iter,
            'feature_count': self.valid_feature_counts,
            'threshold': self.thresholds,
        }
        logger.debug(self.gain_dfs[self.best_iteration])
        z = pd.DataFrame(data=data_dict)
        # logger.debug(z)
        z_sorted = z.sort_values(by=['score'], ascending=False)
        # logger.debug(z_sorted)

    # TODO: CV5 instead of holdout? Should be better
    # TODO: Add holdout here, it is overfitting with Logistic Regression
    def tune(self, X, y, X_val, y_val, X_holdout, y_holdout, total_runs=999):
        objective_goal_is_negative = False  # Fixed to false if using sklearn scorers # self.model_base.problem_type == REGRESSION  # TODO: if objective function goal = lower (logloss, MAE, etc.)
        logger.log(15, 'Feature-pruning '+str(self.model_base.name)+' for '+str(total_runs)+' runs...')

        if len(self.features_in_iter) == 0:
            valid_features = X.columns.values
        else:
            valid_features = self.features_in_iter[-1]

        iter_since_best = 0
        iter_start = self.cur_iteration
        for iteration in range(self.cur_iteration, total_runs):
            self.cur_iteration = iteration
            logger.debug('iteration: %s ' % iteration)
            X_subset = X[valid_features].copy()
            X_val_subset = X_val[valid_features].copy()
            self.thresholds.append(self.threshold)
            self.valid_feature_counts.append(len(valid_features))
            self.features_in_iter.append(valid_features)

            if self.is_fit and (iteration == iter_start):
                model_iter = self.model_base
            else:
                model_iter = copy.deepcopy(self.model_base)
                model_iter.fit(X=X_subset, y=y, X_val=X_val_subset, y_val=y_val)

            banned_features = []

            feature_importance = None
            if hasattr(model_iter.model, 'feature_importances_'):
                feature_importance = model_iter.model.feature_importances_
            elif hasattr(model_iter.model, 'feature_importance'):
                feature_importance = model_iter.model.feature_importance()

            if feature_importance is not None:
                a = pd.Series(data=feature_importance, index=X_subset.columns)
                unused_features = a[a == 0]

                logger.log(15, 'Unused features after pruning: '+ str(list(unused_features.index)))
                features_to_use = [feature for feature in valid_features if feature not in unused_features.index]
                banned_features += list(unused_features.index)
            else:
                features_to_use = list(valid_features)

            cur_score_val = model_iter.score(X=X_val_subset, y=y_val)
            cur_score = model_iter.score(X=X_holdout[valid_features], y=y_holdout)

            logger.log(15, 'Iter '+str(iteration)+'  Score: '+str(cur_score))
            logger.log(15, 'Iter '+str(iteration)+ '  Score Val: '+str(cur_score_val))
            if objective_goal_is_negative:
                cur_score = -cur_score

            if self.best_score == 0 or cur_score > self.best_score:
                logger.log(15, "New best score found!")
                logger.log(15, str(cur_score)+ ' > '+str(self.best_score))
                self.best_score = cur_score
                self.best_iteration = iteration
                iter_since_best = 0
            else:
                iter_since_best += 1

            self.score_in_iter.append(cur_score)

            if iter_since_best >= self.early_stopping_rounds:
                logger.log(15, "Early stopping on iter %s" % iteration)
                logger.log(15, "Best Iteration: %s" % self.best_iteration)
                self.tuned = True
                break

            gain_df = model_iter.compute_feature_importance(X=X_val_subset, y=y_val, features=features_to_use)
            if not objective_goal_is_negative:
                gain_df = -gain_df

            self.gain_dfs.append(gain_df)

            # TODO: Save gain_df, banned_features

            self.threshold = self.adjust_threshold(gain_df, self.threshold)

            banned_df = gain_df.loc[gain_df >= self.threshold].index
            banned_features += list(banned_df)

            logger.log(15, "Banned features: "+str(banned_features))
            self.banned_features_in_iter.append(banned_features)
            valid_features = [feature for feature in valid_features if feature not in banned_features]

            if len(valid_features) == 0:
                logger.log(15, 'No more features to remove, Feature pruning complete')
                logger.log(15, "score_in_iter: "+str(self.score_in_iter))
                self.tuned = True
                break
        self.tuned = True

    @staticmethod
    def adjust_threshold(gain_df, threshold):
        if gain_df.shape[0] == 0:
            raise BaseException('ran out of features to prune!')
        banned_df = gain_df.loc[gain_df >= threshold].index
        banned_features = list(banned_df)

        if len(banned_features) == 0:
            if threshold < -100000000:  # FIXME: Hacked for regression
                raise BaseException('threshold already max!')
            elif threshold > 0.000001:
                threshold_new = threshold / 2
            elif threshold > 0:
                threshold_new = 0
            elif threshold == 0:
                threshold_new = -0.00000001
            elif (threshold < 0) and (threshold > -0.0001):
                threshold_new = threshold * 2
            else:
                threshold_new = gain_df.max()

            logger.log(10, 'Adjusting threshold to %s' % threshold_new)
            return FeaturePruner.adjust_threshold(gain_df=gain_df, threshold=threshold_new)
        else:
            return threshold
