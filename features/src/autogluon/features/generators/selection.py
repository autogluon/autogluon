from .abstract import AbstractFeatureGenerator

class SpearmanFeatureSelector(AbstractFeatureGenerator):
    """Select features based on Spearman correlation with the target.

    Parameters
    ----------
    threshold : float, default=0.1
        Features with absolute Spearman correlation below this threshold will be removed.
    """

    def __init__(self, target_type: str, threshold: float = None, max_features: int = 2000, random_state=42, **kwargs):
        super().__init__(**kwargs)
        if threshold is None and max_features is None:
            raise ValueError("Either 'threshold' or 'max_features' must be provided.")
        self.target_type = target_type
        self.random_state = random_state
        self.threshold = threshold
        self.max_features = max_features
        self.selected_features_ = []

    def _fit(self, X, y):
        # Compute Spearman correlation
        # TODO: Add nan filtering
        # TODO: Add option for AUC for binary
        # TODO: Properly handle multiclass targets
        # X.columns[X.isna().mean()<0.99]
        corr = X.corrwith(y, method='spearman')
        abs_corr = corr.abs().sort_values(ascending=False).dropna()

        if self.threshold is not None:
            self.selected_features_ = abs_corr[abs_corr >= self.threshold].index.tolist()
        elif self.max_features is not None:
            self.selected_features_ = abs_corr.head(self.max_features).index.tolist()

    def _transform(self, X):
        return X[self.selected_features_]
    
    def _fit_transform(self, X, y, **kwargs):
        self._fit(X, y)
        return self._transform(X), dict()
    
    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()