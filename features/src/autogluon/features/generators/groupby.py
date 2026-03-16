import numpy as np
import pandas as pd

from autogluon.common.features.types import R_CATEGORY, R_OBJECT

from .abstract import AbstractFeatureGenerator


# ----------------------------
# Aggregations
# ----------------------------
def q25(series):
    return series.quantile(0.25)


def q75(series):
    return series.quantile(0.75)


def q10(series):
    return series.quantile(0.10)


def q90(series):
    return series.quantile(0.90)


AGGREGATION_REGISTRY = {
    "mean": {"kind": "group", "agg": "mean"},
    "std": {"kind": "group", "agg": "std"},
    "median": {"kind": "group", "agg": "median"},
    "count": {"kind": "group", "agg": "count"},
    "nunique": {"kind": "group", "agg": pd.Series.nunique},
    "min": {"kind": "group", "agg": "min"},
    "max": {"kind": "group", "agg": "max"},
    "q10": {"kind": "group", "agg": q10},
    "q25": {"kind": "group", "agg": q25},
    "q75": {"kind": "group", "agg": q75},
    "q90": {"kind": "group", "agg": q90},
    "pct_rank": {"kind": "rowwise"},
}


def rank_categoricals_by_small_counts(
    X: pd.DataFrame,
    categorical_cols,
    min_count: int = 1,
    top_k_smallest: int = 10,
    require_at_least_levels: int = 2,
    observed: bool = True,
):
    """
    Returns categorical_cols sorted best->worst by lexicographic comparison of the
    smallest group sizes (min, 2nd-min, ...).

    Score vector per cat:
      v = sorted(counts[counts >= min_count])[:top_k_smallest]
    Pad with +inf to fixed length so fewer levels doesn't get penalized.
    Sort by v descending lexicographically.
    """
    scores = {}
    for cat in categorical_cols:
        counts = X[cat].value_counts(dropna=True)
        counts = counts[counts >= min_count].sort_values()  # ascending

        if len(counts) < require_at_least_levels:
            v = np.full(top_k_smallest, -np.inf, dtype=float)
        else:
            v = counts.to_numpy(dtype=float)[:top_k_smallest]
            if v.size < top_k_smallest:
                v = np.pad(v, (0, top_k_smallest - v.size), constant_values=np.inf)

        scores[cat] = v

    ranked = sorted(scores.keys(), key=lambda c: tuple(scores[c]), reverse=True)
    return ranked, scores


class GroupByFeatureGenerator(AbstractFeatureGenerator):
    """
    Output-identical, faster transform:
      - Preallocates output 2D array + builds DataFrame once
      - Caches numeric arrays once
      - Keeps EXACT category-to-code mapping semantics (Index.get_indexer on raw arrays)
      - Keeps EXACT feature insertion order as the original dict-based transform
      - Keeps pct_rank semantics identical (pct_rank is output whenever requested, even if drop_basic=True)
    """

    def __init__(
        self,
        target_type=None,
        aggregations=(
            "mean",
            "pct_rank",
        ),
        relative_to_aggs=("mean",),
        relative_ops=("ratio",),
        drop_basic_groupby_when_relative=True,
        fill_value="nan",
        eps=1e-8,
        return_dataframe=True,
        num_as_cat_cardinality_thresh=2,
        min_num_cardinality_thresh=10,
        max_features=500,
        random_state=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_type = target_type
        self.relative_to_aggs = relative_to_aggs
        self.relative_ops = relative_ops
        self.aggregations = aggregations
        self.max_features = max_features
        self.random_state = random_state

        self.drop_basic_groupby_when_relative = drop_basic_groupby_when_relative

        self.fill_value = np.nan if fill_value == "nan" else fill_value
        self.eps = eps
        self.return_dataframe = return_dataframe

        self.num_as_cat_cardinality_thresh = num_as_cat_cardinality_thresh
        self.min_num_cardinality_thresh = min_num_cardinality_thresh

        unknown = set(self.aggregations) - set(AGGREGATION_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown aggregations: {unknown}")

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        raise ValueError("Input must be a pandas DataFrame")

    def _split_aggs(self):
        group_aggs, rowwise_aggs = [], []
        for name in self.aggregations:
            entry = AGGREGATION_REGISTRY[name]
            if entry["kind"] == "group":
                group_aggs.append(name)
            elif entry["kind"] == "rowwise":
                rowwise_aggs.append(name)
            else:
                raise ValueError(f"Unknown agg kind for {name}: {entry}")
        return group_aggs, rowwise_aggs

    def _relative_enabled(self) -> bool:
        return ("diff" in self.relative_ops) or ("ratio" in self.relative_ops)

    def _drop_basic(self) -> bool:
        # Same logic as your original
        return bool(self.drop_basic_groupby_when_relative and self._relative_enabled() and self.relative_to_aggs)

    def _features_per_pair(self):
        """
        IMPORTANT: kept identical to your original implementation for pair selection / budgeting.
        (Yes, this means pct_rank is NOT counted when drop_basic=True, matching your current behavior.)
        """
        group_aggs, rowwise_aggs = self._split_aggs()

        base = 0
        if not self._drop_basic():
            base += len(group_aggs)
            base += int("pct_rank" in rowwise_aggs)

        rel = 0
        if self._relative_enabled():
            rel += len(self.relative_to_aggs) * (int("diff" in self.relative_ops) + int("ratio" in self.relative_ops))
        return int(base + rel)

    # ----------------------------
    # FIT
    # ----------------------------
    def _fit(self, X, y=None):
        X = self._to_dataframe(X)

        # infer types
        self.categorical_features = X.columns[X.nunique() < self.num_as_cat_cardinality_thresh].tolist()
        self.categorical_features += X.select_dtypes(include=[R_CATEGORY, R_OBJECT]).columns.tolist()
        self.categorical_features = np.unique(self.categorical_features).tolist()

        self.numeric_features = [
            col
            for col in X.columns
            if col not in self.categorical_features
            and X[col].dtype not in ["category", "object"]
            and X[col].nunique() >= self.min_num_cardinality_thresh
        ]

        if len(self.categorical_features) == 0 or len(self.numeric_features) == 0:
            self.group_stats_ = {}
            self.group_index_ = {}
            self.group_values_ = {}
            self.pct_rank_keys_ = {}
            self.pct_rank_vals_ = {}
            self.pct_rank_by_code_ = {}
            self.global_stats_ = {}
            self.output_columns_ = []
            self.pairs_ = []
            return self

        group_aggs, rowwise_aggs = self._split_aggs()
        drop_basic = self._drop_basic()

        ranked_cats, _ = rank_categoricals_by_small_counts(
            X,
            categorical_cols=self.categorical_features,
            min_count=20,
            top_k_smallest=10,
        )
        ranked_nums = X[self.numeric_features].nunique().sort_values(ascending=False).index.to_list()

        self.group_stats_ = {}
        self.group_index_ = {}
        self.group_values_ = {}
        self.pct_rank_keys_ = {}
        self.pct_rank_vals_ = {}
        self.pct_rank_by_code_ = {}  # keep same lazy cache behavior, but initialize dict
        self.global_stats_ = {num: float(X[num].mean()) for num in self.numeric_features}

        # NEW: store exact transform iteration order + exact output column order
        self.pairs_ = []
        self.output_columns_ = []

        features_per_pair = self._features_per_pair()
        budget = self.max_features if self.max_features is not None else float("inf")
        used_features = 0

        for cat in ranked_cats:
            if cat not in self.categorical_features:
                continue

            for num in ranked_nums:
                if num not in self.numeric_features:
                    continue

                if used_features + features_per_pair > budget:
                    return self

                # group-level stats
                if group_aggs:
                    named_aggs = {
                        name: pd.NamedAgg(column=num, aggfunc=AGGREGATION_REGISTRY[name]["agg"]) for name in group_aggs
                    }
                    stats = X.groupby(cat, observed=True).agg(**named_aggs).astype(float)
                else:
                    stats = pd.DataFrame(index=X[cat].dropna().unique())

                self.group_stats_[(cat, num)] = stats

                idx = stats.index
                self.group_index_[(cat, num)] = idx
                self.group_values_[(cat, num)] = {
                    agg: stats[agg].to_numpy(dtype=float, copy=False) for agg in stats.columns
                }

                if "pct_rank" in rowwise_aggs:
                    g = X[[cat, num]].dropna()
                    s = g.groupby(cat, observed=True)[num].apply(
                        lambda t: np.sort(t.to_numpy(dtype=float, copy=False))
                    )
                    self.pct_rank_keys_[(cat, num)] = s.index.to_numpy()
                    self.pct_rank_vals_[(cat, num)] = s.to_numpy()  # dtype=object

                # record order (matches original dict insertion order)
                self.pairs_.append((cat, num))

                # EXACT output layout:
                # - group aggs only when not drop_basic
                if not drop_basic:
                    for agg in group_aggs:
                        self.output_columns_.append(f"{num}__by__{cat}__{agg}")

                # - pct_rank ALWAYS when requested (matches your original transform)
                if "pct_rank" in rowwise_aggs:
                    self.output_columns_.append(f"{num}__by__{cat}__pct_rank")

                # - relatives in the same nested loop order as original
                if self._relative_enabled():
                    for agg in self.relative_to_aggs:
                        if "diff" in self.relative_ops:
                            self.output_columns_.append(f"{num}__minus__by__{cat}_{agg}")
                        if "ratio" in self.relative_ops:
                            self.output_columns_.append(f"{num}__ratio__by__{cat}_{agg}")

                used_features += features_per_pair

        return self

    # ----------------------------
    # TRANSFORM
    # ----------------------------
    def _transform(self, X):
        X = self._to_dataframe(X)

        if len(getattr(self, "pairs_", [])) == 0:
            empty = pd.DataFrame(index=X.index)
            return empty if self.return_dataframe else empty.values

        group_aggs, rowwise_aggs = self._split_aggs()
        drop_basic = self._drop_basic()
        rel_enabled = self._relative_enabled()

        n = len(X)
        m = len(self.output_columns_)
        out = np.empty((n, m), dtype=float)
        col_i = 0

        # cache numeric arrays once
        used_nums = {num for _, num in self.pairs_}
        num_cache = {num: X[num].astype(float).to_numpy(copy=False) for num in used_nums}

        # cache category codes once per cat (EXACT original semantics)
        codes_cache = {}  # cat -> np.ndarray[int]
        missing_cache = {}  # cat -> np.ndarray[bool]
        safe_cache = {}  # cat -> np.ndarray[int] with -1 replaced by 0 for take()

        for cat, num in self.pairs_:
            x = num_cache[num]

            idx = self.group_index_.get((cat, num), None)
            vals_dict = self.group_values_.get((cat, num), None)

            mapped = {}

            # compute codes once per cat, using the FIRST idx encountered for that cat
            if cat not in codes_cache:
                if idx is None:
                    codes = np.full(n, -1, dtype=int)
                else:
                    cat_arr = X[cat].to_numpy()
                    codes = idx.get_indexer(cat_arr)  # exact same as your original
                codes_cache[cat] = codes
                missing = codes == -1
                missing_cache[cat] = missing

                safe_codes = codes.copy()
                safe_codes[missing] = 0
                safe_cache[cat] = safe_codes

            codes = codes_cache[cat]
            missing = missing_cache[cat]
            safe_codes = safe_cache[cat]

            # ---- group aggs mapping ----
            if idx is not None and vals_dict is not None and len(vals_dict) > 0:
                for agg, vals in vals_dict.items():
                    col = vals.take(safe_codes).astype(float, copy=False)
                    if missing.any():
                        col = col.copy()
                        col[missing] = np.nan

                    mapped[agg] = col

                    if not drop_basic:
                        if self.fill_value is np.nan:
                            out[:, col_i] = col
                        else:
                            out[:, col_i] = np.where(np.isnan(col), self.fill_value, col)
                        col_i += 1
            else:
                # fallback identical to your original
                stats = self.group_stats_.get((cat, num), None)
                if stats is not None and len(stats.columns) > 0:
                    cat_series = X[cat]
                    for agg in stats.columns:
                        col = cat_series.map(stats[agg]).astype(float).to_numpy(copy=False)
                        mapped[agg] = col
                        if not drop_basic:
                            if self.fill_value is np.nan:
                                out[:, col_i] = col
                            else:
                                out[:, col_i] = np.where(np.isnan(col), self.fill_value, col)
                            col_i += 1

            # ---- pct_rank: ALWAYS output when requested (matches your original) ----
            if "pct_rank" in rowwise_aggs:
                pr = np.full(n, 0.5, dtype=float)
                vals = x

                valid = (codes != -1) & (~np.isnan(vals))
                if valid.any():
                    keys = self.pct_rank_keys_.get((cat, num), None)
                    dists = self.pct_rank_vals_.get((cat, num), None)

                    if idx is not None and keys is not None and dists is not None and len(keys) > 0:
                        dist_by_code = self.pct_rank_by_code_.get((cat, num), None)
                        if dist_by_code is None:
                            dist_by_code = np.empty(len(idx), dtype=object)
                            dist_by_code[:] = None
                            pos = idx.get_indexer(keys)
                            ok = pos != -1
                            dist_by_code[pos[ok]] = dists[ok]
                            self.pct_rank_by_code_[(cat, num)] = dist_by_code

                        vidx = np.flatnonzero(valid)
                        vcode = codes[vidx]
                        order = np.argsort(vcode, kind="mergesort")
                        vidx = vidx[order]
                        vcode = vcode[order]

                        breaks = np.r_[0, 1 + np.flatnonzero(vcode[1:] != vcode[:-1]), vcode.size]
                        for b0, b1 in zip(breaks[:-1], breaks[1:]):
                            c = vcode[b0]
                            arr = dist_by_code[c]
                            if arr is None or arr.size == 0:
                                continue
                            rows = vidx[b0:b1]
                            ranks = np.searchsorted(arr, vals[rows], side="right")
                            pr[rows] = ranks / arr.size

                out[:, col_i] = pr
                col_i += 1

            # ---- relatives ----
            if rel_enabled:
                missing_aggs = set(self.relative_to_aggs) - set(mapped)
                if missing_aggs:
                    raise ValueError(
                        f"Requested relative_to_aggs {missing_aggs} not present in computed "
                        f"group aggregations {list(mapped)} for pair ({cat}, {num}). "
                        f"Make sure those aggs are included in `aggregations=`."
                    )

                for agg in self.relative_to_aggs:
                    ref = mapped[agg]
                    if "diff" in self.relative_ops:
                        out[:, col_i] = x - ref
                        col_i += 1
                    if "ratio" in self.relative_ops:
                        out[:, col_i] = x / (ref + self.eps)
                        col_i += 1

        if col_i != m:
            raise RuntimeError(
                f"Internal error: wrote {col_i} columns but expected {m}. "
                f"This indicates a mismatch between output_columns_ and transform writing order."
            )

        result = pd.DataFrame(out, columns=self.output_columns_, index=X.index)
        return result if self.return_dataframe else result.values

    def _fit_transform(self, X, y, **kwargs):
        self._fit(X, y)
        return self._transform(X), dict()

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()
