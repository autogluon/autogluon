"""
Code adapted from skrub==0.6.2
"""


from __future__ import annotations

import sklearn
from sklearn.utils.fixes import parse_version

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)


if sklearn_version < parse_version("1.6"):
    def validate_data(_estimator, /, **kwargs):
        if "ensure_all_finite" in kwargs:
            force_all_finite = kwargs.pop("ensure_all_finite")
        else:
            force_all_finite = True
        return _estimator._validate_data(**kwargs, force_all_finite=force_all_finite)
else:
    from sklearn.utils.validation import validate_data  # noqa: F401
