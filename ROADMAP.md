# AutoGluon Roadmap 2022

In 2022, our top priorities are:

- [x] (v0.4) Windows OS Support
- [x] (v0.4) Python 3.9 Support
- [ ] Time-Series Support
- [x] (v0.4) HuggingFace Integration
- [ ] Improved Multi-modal Modeling
- [ ] Advance SOTA in Stack Ensemble Research (Deeper Stack Ensembles)
- [x] (v0.4) Cloud Training & Deployment
- [x] (v0.4) Parallel Model Training
- [ ] Parallel Hyperparameter Tuning
- [ ] Distributed Model Training
- [ ] Distributed Hyperparameter Tuning
- [x] (v0.4) Semi-supervised Learning
- [x] (v0.4) Automated Model Calibration via Temperature Scaling
- [ ] Enhanced Model Distillation
- [x] (v0.4) Online Inference Optimization
- [ ] Meta-Learning
- [ ] Improved Large-scale Data Handling (10M+ Rows)
- [ ] Improved Feature Type Inference
- [ ] Improved Feature Preprocessing
- [ ] Refactor autogluon.features into Standalone Module
- [ ] Covariate Shift Detection
- [ ] Covariate Shift Correction
- [ ] Exploratory Analysis
- [ ] Model Interpretability
- [ ] Model Uncertainty
- [ ] Model Monitoring
- [ ] Model Calibration (Conformal Methods)
- [ ] Image Model Inference Optimization
- [ ] Text Model Inference Optimization
- [ ] Advanced Custom Model Tutorial

## AutoGluon 0.4 Features, Released: March 2022

[v0.4 Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.4.0)

### Themes and Major Features

- [x] Windows OS Support
- [x] Python 3.9 Support
- [x] HuggingFace Integration
- [x] Torch Migration (Remove MXNet dependency)
- [x] Parallel Model Training (2x training speed-up for bagging/stacking)
- [x] Automated Feature Pruning/Selection
- [x] Semi-supervised & Transductive Learning Support
- [x] Automated Model Calibration via Temperature Scaling
- [x] Cloud Training & Deployment Tutorials
- [x] Feature Preprocessing Tutorial
- [x] Documentation Overhaul
- [x] Hyperparameter Tuning Overhaul
- [x] Memory Usage Optimizations
- [x] Various Performance Optimizations
- [x] Various Bug Fixes

# AutoGluon Roadmap 2021

In 2021, our top priorities are:

- [x] Make AutoGluon the most versatile AutoML framework via dedicated multi-modal image-text-tabular support ([paper](https://arxiv.org/abs/2111.02705)).
- [x] Modularization of the various components of AutoGluon.
- [x] Model Training Speed Optimizations.
- [x] Model Inference Speed Optimizations.
- [x] Model Quality Optimizations.
- [x] Integration with [NVIDIA RAPIDS](https://developer.nvidia.com/rapids) for accelerated GPU training.
- [x] Integration with [Intel sklearnex](https://github.com/intel/scikit-learn-intelex) for accelerated CPU training.
- [x] Improved documentation and tutorials.
- [x] Training and Inference containers.

### AutoGluon 0.3.1 Features, Released: August 2021

[Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.3.1)

### AutoGluon 0.3 Features, Released: August 2021

[Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.3.0)

### AutoGluon 0.2 Features, Released: April 2021

[Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.2.0)

### AutoGluon 0.1 Features, Released: March 2021

[Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.1.0)

# AutoGluon Roadmap 2020

In 2020, we plan to focus on improving code quality, extensibility, and robustness of the package.

We will work towards unifying the APIs of the separate tasks (Tabular, Image, Text) to simplify and streamline development and improve the user experience.

### 2020 Releases

- [v0.0.15 Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.0.15) (December 2020)
- [v0.0.14 Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.0.14) (October 2020, Highlight: Added FastAI Neural Network Model)
- [v0.0.13 Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.0.13) (August 2020, Highlight: Added model distillation ([paper](https://arxiv.org/abs/2006.14284)))
- [v0.0.12 Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.0.12) (July 2020, Highlight: Added custom model support)
- [v0.0.11 Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.0.11) (June 2020)
- [v0.0.10 Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.0.10) (June 2020, Highlight: Implemented feature importance)
- [v0.0.9 Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.0.9) (May 2020)
- [v0.0.8 Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.0.8) (May 2020)
- [v0.0.7 Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.0.7) (May 2020, Highlight: first addition of the `presets` argument)
- [v0.0.6 Release Notes](https://github.com/awslabs/autogluon/releases/tag/v0.0.6) (March 2020, first release tagged on GitHub with release notes)
- [v0.0.5 Release](https://pypi.org/project/autogluon/0.0.5/) (February 2020, used in the original [AutoGluon-Tabular paper](https://arxiv.org/abs/2003.06505))
- [v0.0.4 Release](https://pypi.org/project/autogluon/0.0.4/) (January 2020)

# AutoGluon Roadmap 2019

In 2019, we plan to release the initial open source version of AutoGluon, featuring Tabular, Text, and Image classification and regression tasks, along with Object Detection.

### 2019 Releases

- [v0.0.3 Release](https://pypi.org/project/autogluon/0.0.3/) (December 2019)
- [v0.0.2 Release](https://pypi.org/project/autogluon/0.0.2/) (December 2019, Initial Open Source Release)
