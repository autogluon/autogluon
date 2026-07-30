The code in this directory is adapted from the original repository: https://github.com/DataDog/toto,
specifically the `toto2` and `dd_unit_scaling` packages. It has been reduced to the inference-only
subset that AutoGluon uses, with minor edits for code conventions. Training code, the GluonTS
predictor, multivariate/covariate support and the distributed unit scaling machinery were removed.
The `unit_scaling.py` module vendors the u-muP primitives that `dd_unit_scaling` and `unit_scaling`
provide, so that neither package is required at runtime. The `unit_scaling` package here refers to
graphcore-research/unit-scaling (https://github.com/graphcore-research/unit-scaling); `unit_scaling.py`
therefore includes code derived from that project.

Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.

This product includes software developed at Datadog (https://www.datadoghq.com/)
Copyright 2026 Datadog, Inc.

The `unit_scaling.py` module also includes code derived from graphcore-research/unit-scaling
(https://github.com/graphcore-research/unit-scaling), licensed under the Apache-2.0 License.
Copyright 2023 Graphcore Ltd.
