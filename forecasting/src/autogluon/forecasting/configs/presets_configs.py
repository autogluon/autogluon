forecasting_presets_configs = dict(
    high_quality={"hyperparameters": "default_hpo",
                  "hyperparameter_tune_kwargs": {'scheduler': 'local',
                                                'searcher': 'random',
                                                'num_trials': 10
                                                },
                  "refit_full": True
                  },
    good_quality={"hyperparameters": "default_hpo",
                  "hyperparameter_tune_kwargs": {'scheduler': 'local',
                                                'searcher': 'random',
                                                'num_trials': 2
                                                },
                  "refit_full": True
                  },
    medium_quality={"hyperparameters": "default",
                    "refit_full": True},
    low_quality={"hyperparameters": "toy"},
    low_quality_hpo={"hyperparameters": "toy_hpo",
                      "hyperparameter_tune_kwargs": {'scheduler': 'local',
                                                     'searcher': 'random',
                                                     'num_trials': 2
                                                     },
                      "refit_full": True
                    },

)
