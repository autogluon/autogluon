from PIL import Image
import requests
import pytest
from autogluon.multimodal import MultiModalPredictor


def test_clip_zero_shot():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image_name = "clip_image_0.jpg"
    image.save(image_name)

    predictor = MultiModalPredictor(hyperparameters={"model.names": ["clip"]}, problem_type="zero_shot")

    pred = predictor.predict({"image": [image_name], "text": ["a photo of a dog"]})
    assert pred.shape == (1,)

    pred = predictor.predict({"image": [image_name, image_name], "text": ["a photo of a dog", "a photo of a cat"]})
    assert pred.shape == (2,)

    pred = predictor.predict({"image": [image_name, image_name]}, {"text": ["a photo of a dog", "a photo of a cat"]})
    assert pred.shape == (2,)

    prob = predictor.predict_proba({"image": [image_name]}, {"text": ["a photo of a dog", "a photo of a cat"]})
    assert prob.shape == (1, 2)
    for per_row_prob in prob:
        assert pytest.approx(sum(per_row_prob), 1e-6) == 1

    prob = predictor.predict_proba({"image": [image_name, image_name]}, {"text": ["a photo of a dog", "a photo of a cat"]})
    assert prob.shape == (2, 2)
    for per_row_prob in prob:
        assert pytest.approx(sum(per_row_prob), 1e-6) == 1

    embedding = predictor.extract_embedding({"image": [image_name]})
    assert list(embedding.keys()) == ["image"]
    for v in embedding.values():
        assert v.shape == (1, 512)

    embedding = predictor.extract_embedding({"text": ["a photo of a dog"]})
    assert list(embedding.keys()) == ["text"]
    for v in embedding.values():
        assert v.shape == (1, 512)

    embedding = predictor.extract_embedding({"image": [image_name], "text": ["a photo of a dog"]})
    assert list(embedding.keys()).sort() == ["image", "text"].sort()
    for v in embedding.values():
        assert v.shape == (1, 512)
