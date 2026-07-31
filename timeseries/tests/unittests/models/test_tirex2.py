import numpy as np
import pytest

from autogluon.timeseries.models import TiRex2Model
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator

from ..common import (
    get_data_frame_with_covariates,
    get_data_frame_with_item_index,
    get_data_frame_with_variable_lengths,
)
from .common import DEVICE_TEST_CASES, TIREX2_MODEL_PATH, tirex2_available

pytestmark = pytest.mark.skipif(not tirex2_available(), reason="tirex-2 is not installed")


def get_tirex2_model(prediction_length=3, quantile_levels=(0.1, 0.5, 0.9), covariate_metadata=None, **hyperparameters):
    hyperparameters = {"model_path": TIREX2_MODEL_PATH, "device": "cpu", **hyperparameters}
    return TiRex2Model(
        prediction_length=prediction_length,
        freq="D",
        quantile_levels=list(quantile_levels),
        covariate_metadata=covariate_metadata,
        hyperparameters=hyperparameters,
    )


class TestTiRex2Interpolation:
    def test_when_requested_levels_match_knots_then_interpolation_is_identity(self):
        knots = np.array([0.1, 0.5, 0.9])
        forecast = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        result = TiRex2Model._interpolate_quantiles(forecast, knots, knots)
        np.testing.assert_allclose(result, forecast)

    def test_when_level_between_knots_then_value_is_linearly_interpolated(self):
        knots = np.array([0.1, 0.5, 0.9])
        forecast = np.array([[0.0, 10.0, 20.0]])
        # 0.3 is halfway between knots 0.1 and 0.5 -> halfway between 0.0 and 10.0
        result = TiRex2Model._interpolate_quantiles(forecast, knots, np.array([0.3]))
        np.testing.assert_allclose(result, [[5.0]])

    def test_when_level_outside_knots_then_value_is_clipped_to_extreme_knot(self):
        knots = np.array([0.1, 0.5, 0.9])
        forecast = np.array([[1.0, 2.0, 3.0]])
        result = TiRex2Model._interpolate_quantiles(forecast, knots, np.array([0.01, 0.99]))
        np.testing.assert_allclose(result, [[1.0, 3.0]])


class TestTiRex2Dataset:
    @pytest.mark.parametrize(
        "input_data_length, context_length",
        [(100, 10), (100, 100), (5, 100)],
    )
    def test_when_context_length_set_then_each_series_is_truncated(self, input_data_length, context_length):
        df = get_data_frame_with_item_index(["A", "B", "C", "D"], data_length=input_data_length)
        model = get_tirex2_model()

        timeseries = model._to_timeseries(df, context_length=context_length)

        assert len(timeseries) == 4
        for ts in timeseries:
            assert ts.target.shape[0] == 1  # univariate
            assert ts.target.shape[-1] == min(context_length, input_data_length)

    def test_when_lengths_are_uneven_then_each_series_keeps_its_own_length(self):
        item_id_to_length = {"A": 1, "B": 25, "C": 50, "D": 100}
        df = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)
        model = get_tirex2_model()

        timeseries = model._to_timeseries(df, context_length=1000)

        for ts, expected_length in zip(timeseries, item_id_to_length.values()):
            assert ts.target.shape[-1] == expected_length


class TestTiRex2Predict:
    @pytest.mark.parametrize("prediction_length", [1, 5])
    def test_when_predict_then_output_has_expected_shape_and_no_nans(self, prediction_length):
        item_ids = ["A", "B", "C"]
        df = get_data_frame_with_item_index(item_ids, data_length=50)
        model = get_tirex2_model(prediction_length=prediction_length, quantile_levels=(0.1, 0.5, 0.9))
        model.fit(train_data=df)

        predictions = model.predict(df)

        assert list(predictions.columns) == ["mean", "0.1", "0.5", "0.9"]
        assert len(predictions) == len(item_ids) * prediction_length
        assert not predictions.isna().any().any()

    def test_when_series_contain_nans_then_predictions_have_no_nans(self):
        df = get_data_frame_with_item_index(["A", "B"], data_length=50)
        df.iloc[:10] = float("nan")  # leading gap in the first series
        model = get_tirex2_model()
        model.fit(train_data=df)

        predictions = model.predict(df)

        assert not predictions.isna().any().any()

    def test_when_series_is_all_nan_then_predictions_have_no_nans(self):
        df = get_data_frame_with_item_index(["A", "B"], data_length=50)
        df["target"] = float("nan")
        model = get_tirex2_model()
        model.fit(train_data=df)

        predictions = model.predict(df)

        assert not predictions.isna().any().any()


class TestTiRex2Covariates:
    # NOTE: end-to-end fit+predict with real & categorical, past & known covariates is already covered for every
    # covariate-supporting model by `test_models.py::test_when_covariate_regressor_is_used_then_model_can_fit_and_predict`.
    # These tests only cover the wrapper-specific `_to_timeseries` covariate assembly.
    def test_when_covariates_present_then_real_and_encoded_cat_are_passed_with_correct_shapes(self):
        prediction_length = 5
        context_length = 1000
        known_covariates_names = ["known_real", "known_cat"]
        raw = get_data_frame_with_covariates(
            item_id_to_length={"A": 40, "B": 55, "C": 30},
            covariates_real=["known_real", "past_real"],
            covariates_cat=["known_cat", "past_cat"],
        )
        feature_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
        data = feature_generator.fit_transform(raw)
        model = get_tirex2_model(
            prediction_length=prediction_length, covariate_metadata=feature_generator.covariate_metadata
        )
        past_data, known_covariates = data.get_model_inputs_for_scoring(prediction_length, known_covariates_names)

        timeseries = model._to_timeseries(past_data, context_length=context_length, known_covariates=known_covariates)

        indptr = past_data.get_indptr()
        for idx, ts in enumerate(timeseries):
            context_len = indptr[idx + 1] - indptr[idx]
            # One real (passed through) + one target-encoded categorical covariate in each of the past and known groups.
            assert ts.past_covariates.shape == (2, context_len)
            assert ts.future_covariates.shape == (2, context_len + prediction_length)

    def test_when_categorical_known_covariate_then_it_is_target_encoded_consistently(self):
        # Build a known categorical that perfectly determines the target, so target encoding maps each category to its
        # own (item-level) target mean. The same category must therefore encode to the same value everywhere it
        # appears, including across the context/horizon boundary.
        prediction_length = 3
        known_covariates_names = ["known_cat"]
        raw = get_data_frame_with_covariates(item_id_to_length={"A": 12})
        categories = ["foo", "bar", "baz"] * 4
        raw["known_cat"] = categories
        # Target is a deterministic function of the category, so each category has a distinct encoded value.
        cat_to_value = {"foo": 10.0, "bar": 20.0, "baz": 30.0}
        raw["target"] = [cat_to_value[c] for c in categories]
        feature_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
        data = feature_generator.fit_transform(raw)
        model = get_tirex2_model(
            prediction_length=prediction_length, covariate_metadata=feature_generator.covariate_metadata
        )
        past_data, known_covariates = data.get_model_inputs_for_scoring(prediction_length, known_covariates_names)

        timeseries = model._to_timeseries(past_data, context_length=1000, known_covariates=known_covariates)
        # future_covariates holds only the known-covariate rows; the single known cat is row 0.
        encoded = timeseries[0].future_covariates[0]
        cat_sequence = past_data["known_cat"].tolist() + known_covariates["known_cat"].tolist()

        # Every occurrence of the same category encodes to the same value, and distinct categories differ.
        value_by_category = {}
        for category, value in zip(cat_sequence, encoded.tolist()):
            value_by_category.setdefault(category, value)
            assert value == pytest.approx(value_by_category[category])
        assert len(set(round(v, 6) for v in value_by_category.values())) == 3

    def test_when_target_encoding_disabled_then_categorical_is_ordinal_encoded(self):
        # With use_target_encoding=False, categories map to their integer codes (0, 1, 2, ...) rather than target means.
        prediction_length = 3
        known_covariates_names = ["known_cat"]
        raw = get_data_frame_with_covariates(item_id_to_length={"A": 12})
        raw["known_cat"] = ["foo", "bar", "baz"] * 4
        feature_generator = TimeSeriesFeatureGenerator(target="target", known_covariates_names=known_covariates_names)
        data = feature_generator.fit_transform(raw)
        model = get_tirex2_model(
            prediction_length=prediction_length,
            covariate_metadata=feature_generator.covariate_metadata,
            use_target_encoding=False,
        )
        past_data, known_covariates = data.get_model_inputs_for_scoring(prediction_length, known_covariates_names)

        timeseries = model._to_timeseries(
            past_data, context_length=1000, known_covariates=known_covariates, use_target_encoding=False
        )
        encoded = timeseries[0].future_covariates[0]
        expected_codes = past_data["known_cat"].cat.codes.tolist() + known_covariates["known_cat"].cat.codes.tolist()
        assert encoded.tolist() == pytest.approx([float(c) for c in expected_codes])


class TestTiRex2Device:
    @pytest.mark.parametrize("device_arg, cuda_available, expected_device", DEVICE_TEST_CASES)
    def test_when_device_requested_then_expected_device_is_resolved(
        self, device_arg, cuda_available, expected_device, monkeypatch
    ):
        model = get_tirex2_model(device=device_arg)
        monkeypatch.setattr(model, "_is_gpu_available", lambda: cuda_available)
        assert model._get_device() == expected_device
