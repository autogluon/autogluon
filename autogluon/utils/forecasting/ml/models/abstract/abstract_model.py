from gluonts.evaluation.backtest import make_evaluation_predictions
from ....utils.loaders import load_pickle
from ....utils.savers import save_pickle


class AbstractModel:

    def __init__(self, hyperparameters=None, model=None):
        self.set_default_parameters()
        self.params = {}
        if hyperparameters is not None:
            self.params.update(hyperparameters)
        self.model = model

    def save(self, path):
        save_pickle.save(path=path, obj=self)

    @classmethod
    def load(cls, path):
        return load_pickle.load(path)

    def set_default_parameters(self):
        pass

    def create_model(self):
        pass

    def fit(self, train_ds):
        pass
        # self.model = self.model.train(train_ds)

    def predict(self, test_ds, num_samples=100):
        pass

    def hyperparameter_tune(self, train_data, test_data, scheduler_options, **kwargs):
        pass

    def score(self, y, y_true):
        pass

    # def _get_hpo_results(self, scheduler, scheduler_options, time_start):
    #     # Store results / models from this HPO run:
    #     best_hp = scheduler.get_best_config()  # best_hp only contains searchable stuff
    #     hpo_results = {
    #         'best_reward': scheduler.get_best_reward(),
    #         'best_config': best_hp,
    #         # 'total_time': time.time() - time_start,
    #         'metadata': scheduler.metadata,
    #         'training_history': scheduler.training_history,
    #         'config_history': scheduler.config_history,
    #         'reward_attr': scheduler._reward_attr,
    #         # 'args': model_trial.args
    #     }
    #     print(hpo_results)
    # #
    # #     hpo_results = BasePredictor._format_results(hpo_results)  # results summarizing HPO for this model
    # #     if ('dist_ip_addrs' in scheduler_options) and (len(scheduler_options['dist_ip_addrs']) > 0):
    # #         raise NotImplementedError("need to fetch model files from remote Workers")
    # #         # TODO: need to handle locations carefully: fetch these files and put them into self.path directory:
    # #         # 1) hpo_results['trial_info'][trial]['metadata']['trial_model_file']
    # #
    # #     hpo_models = {}  # stores all the model names and file paths to model objects created during this HPO run.
    # #     hpo_model_performances = {}
    # #     for trial in sorted(hpo_results['trial_info'].keys()):
    # #         # TODO: ignore models which were killed early by scheduler (eg. in Hyperband). How to ID these?
    # #         file_id = "trial_" + str(trial)  # unique identifier to files from this trial
    # #         trial_model_name = self.name + os.path.sep + file_id
    # #         trial_model_path = self.path_root + trial_model_name + os.path.sep
    # #         hpo_models[trial_model_name] = trial_model_path
    # #         hpo_model_performances[trial_model_name] = hpo_results['trial_info'][trial][scheduler._reward_attr]
    #
    #     # logger.log(15, "Time for %s model HPO: %s" % (self.name, str(hpo_results['total_time'])))
    #     # logger.log(15, "Best hyperparameter configuration for %s model: " % self.name)
    #     # logger.log(15, str(best_hp))
    #     # return hpo_models, hpo_model_performances, hpo_results
