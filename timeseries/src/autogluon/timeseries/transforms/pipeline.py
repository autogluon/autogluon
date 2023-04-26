from typing import List

from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame

from .abstract_transform import AbstractTransform


class PipelineTransform(AbstractTransform):
    """Combination of multiple transforms as a single operation."""

    def __init__(self, transforms: List[AbstractTransform], target: str = "target"):
        super().__init__(target)
        assert all(isinstance(t, AbstractTransform) for t in transforms)
        self.transforms = transforms

    def _fit(self, data: TimeSeriesDataFrame) -> "AbstractTransform":
        raise NotImplementedError(f"{self.name} should be fit using the fit_transform method.")

    def _transform(self, data: TimeSeriesDataFrame, inplace: bool = False) -> TimeSeriesDataFrame:
        if not inplace:
            data = data.copy()
        for t in self.transforms:
            data = t.transform(data, inplace=True)  # avoid copying data in each transform, copy once outside
        return data

    def fit_transform(self, data: TimeSeriesDataFrame, inplace: bool = False) -> TimeSeriesDataFrame:
        if not inplace and len(self.transforms) > 0:
            data = data.copy()
        for t in self.transforms:
            data = t.fit_transform(data, inplace=True)  # avoid copying data in each transform, copy once outside
        return data
