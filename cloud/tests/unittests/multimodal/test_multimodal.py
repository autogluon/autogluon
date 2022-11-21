import tempfile
import zipfile

from autogluon.cloud import MultiModalCloudPredictor


def test_multimodal_tabular_text(test_helper):
    train_data = "tabular_text_train.csv"
    test_data = "tabular_text_test.csv"
    with tempfile.TemporaryDirectory() as root:
        test_helper.prepare_data(train_data, test_data)
        time_limit = 60

        predictor_init_args = dict(
            label="Sentiment",
        )
        predictor_fit_args = dict(train_data=train_data, time_limit=time_limit)
        cloud_predictor = MultiModalCloudPredictor(
            cloud_output_path="s3://autogluon-cloud-ci/test-multimodal-tabular-text",
            local_output_path="test_multimodal_tabular_text_cloud_predictor",
        )
        cloud_predictor_no_train = MultiModalCloudPredictor(
            cloud_output_path="s3://autogluon-cloud-ci/test-multimodal-tabular-text-no-train",
            local_output_path="test_multimodal_tabular_text_cloud_predictor_no_train",
        )
        test_helper.test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data,
            fit_instance_type="ml.g4dn.2xlarge",
            fit_kwargs=dict(custom_image_uri=test_helper.gpu_training_image),
            deploy_kwargs=dict(custom_image_uri=test_helper.cpu_inference_image),
            predict_kwargs=dict(custom_image_uri=test_helper.cpu_inference_image),
        )


def test_multimodal_tabular_text_image(test_helper):
    train_data = "tabular_text_image_train.csv"
    test_data = "tabular_text_image_test.csv"
    images = "tabular_text_image_images.zip"
    with tempfile.TemporaryDirectory() as root:
        test_helper.prepare_data(train_data, test_data, images)
        with zipfile.ZipFile(images, "r") as zip_ref:
            zip_ref.extractall(".")
        time_limit = 120

        predictor_init_args = dict(
            label="AdoptionSpeed",
        )
        predictor_fit_args = dict(train_data=train_data, time_limit=time_limit)
        cloud_predictor = MultiModalCloudPredictor(
            cloud_output_path="s3://autogluon-cloud-ci/test-multimodal-tabular-text-image",
            local_output_path="test_multimodal_tabular_text_image_cloud_predictor",
        )
        cloud_predictor_no_train = MultiModalCloudPredictor(
            cloud_output_path="s3://autogluon-cloud-ci/test-multimodal-tabular-text-image-no-train",
            local_output_path="test_multimodal_tabular_text_image_cloud_predictor_no_train",
        )
        test_helper.test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data,
            image_path="tabular_text_image_images.zip",
            fit_instance_type="ml.g4dn.2xlarge",
            fit_kwargs=dict(custom_image_uri=test_helper.gpu_training_image),
            deploy_kwargs=dict(custom_image_uri=test_helper.cpu_inference_image),
            predict_real_time_kwargs=dict(test_data_image_column="Images"),
            predict_kwargs=dict(test_data_image_column="Images", custom_image_uri=test_helper.cpu_inference_image),
        )
