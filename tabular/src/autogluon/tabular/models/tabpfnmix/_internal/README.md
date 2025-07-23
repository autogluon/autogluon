The code located in this directory (`autogluon.tabular.models.tabpfnmix._internal`) is based on the code for TabForestPFN: https://github.com/FelixdenBreejen/TabForestPFN, forked when the codebase was on [this commit](https://github.com/FelixdenBreejen/TabForestPFN/tree/53114795d3c96f87348a7ccbb675665e9d3e5243).

The TabForestPFN codebase was originally created as an implementation of the paper "[Why In-Context Learning Transformers are Tabular Data Classifiers](https://arxiv.org/pdf/2405.13396)" by authors Felix den Breejen, Sangmin Bae, Stephen Cha, Se-Young Yun.

The codebase provides convenient functionality surrounding conducting fine-tuning and inference on tabular transformer models, which we leverage in the TabPFNMix model.

For more information on the TabPFNMix model, refer to the HuggingFace model repositories:

1. https://huggingface.co/autogluon/tabpfn-mix-1.0-classifier
2. https://huggingface.co/autogluon/tabpfn-mix-1.0-regressor

The following changes to the original codebase have been made by Nick Erickson (@innixma) and Xiyuan Zhang (@xiyuanzh):

1. Improved model early stopping logic to properly load the best epoch's weights at the end of the fit call.
2. Removed all code that is unused for model fine-tuning and inference. Therefore, all code related to benchmarking and pre-training have been removed.
3. Removed all dependencies unrelated to model fine-tuning and inference.
4. Optimized fine-tuning checkpoints to use in-memory checkpoints.
5. Added CPU support
6. Added torch thread control
7. Added custom metric support
8. Optimized metric calculation to avoid calculating unnecessary metrics
9. Optimized fine-tuning speed by avoiding calculating metrics for train data by default
10. Replacing several pieces of custom functionality with AutoGluon utility equivalents to reduce code duplication
11. Only checkpoint when a new best iteration is found, rather than each iteration
12. Various cosmetic, logging, typing, docstring, and formatting changes
13. Added random seed control for reproducible results
14. Reduced memory usage and disk usage for inference by 5x by deleting unnecessary checkpoint and optimizer objects
15. Added time_limit support
16. Added support for fine-tuning without validation data
17. Added HuggingFace Hub `from_pretrained` support
18. Added regression support
