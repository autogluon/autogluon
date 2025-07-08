import numpy as np
import pytest
import requests
from PIL import Image

from autogluon.multimodal import MultiModalPredictor


def download_sample_images():
    url = "https://autogluon.s3.us-west-2.amazonaws.com/images/automm_test_images/cats.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    cat_image_name = "cat.jpg"
    image.save(cat_image_name)

    url = "https://autogluon.s3.us-west-2.amazonaws.com/images/automm_test_images/dog.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    dog_image_name = "dog.jpg"
    image.save(dog_image_name)

    return cat_image_name, dog_image_name


def test_clip_zero_shot():
    cat_image_name, dog_image_name = download_sample_images()

    cat_text = "a photo of a cat"
    dog_text = "a photo of a dog"
    bird_text = "a photo of a bird"

    predictor = MultiModalPredictor(
        problem_type="zero_shot_image_classification",
    )

    # compute the cosine similarity of one image-text pair.
    pred = predictor.predict({"image": [cat_image_name], "text": [cat_text]})
    assert pred.shape == (1,)

    # compute the cosine similarities of more image and text pairs.
    pred = predictor.predict({"image": [cat_image_name, dog_image_name], "text": [cat_text, dog_text]})
    assert pred.shape == (2,)

    # match images in a given text pool and output the matched text index (starting from 0) for each image.
    pred = predictor.predict({"image0": [cat_image_name, dog_image_name]}, {"names": [dog_text, cat_text, bird_text]})
    assert pred.shape == (2,)

    # match texts in a given image pool and output the matched image index (starting from 0) for each text.
    pred = predictor.predict(
        {"query": [dog_text, cat_text, bird_text]}, {"candidates": [cat_image_name, dog_image_name]}
    )
    assert pred.shape == (3,)

    # predict the probabilities of one image matching several texts.
    prob = predictor.predict_proba({"image": [cat_image_name]}, {"text": [cat_text, dog_text, bird_text]})
    assert prob.shape == (1, 3)
    for per_row_prob in prob:
        assert pytest.approx(sum(per_row_prob), 1e-6) == 1

    # given two or more images, we can get the probabilities of matching each image with a pool of texts.
    prob = predictor.predict_proba(
        {"image": [dog_image_name, cat_image_name]}, {"text": [bird_text, cat_text, dog_text]}
    )
    assert prob.shape == (2, 3)
    for per_row_prob in prob:
        assert pytest.approx(sum(per_row_prob), 1e-6) == 1

    # predict the probabilities of one text matching several images.
    prob = predictor.predict_proba({"x": [cat_text]}, {"y": [dog_image_name, cat_image_name]})
    assert prob.shape == (1, 2)
    for per_row_prob in prob:
        assert pytest.approx(sum(per_row_prob), 1e-6) == 1

    # given two or more texts, we can get the probabilities of matching each text with a pool of images.
    prob = predictor.predict_proba({"a": [bird_text, cat_text, dog_text]}, {"b": [dog_image_name, cat_image_name]})
    assert prob.shape == (3, 2)
    for per_row_prob in prob:
        assert pytest.approx(sum(per_row_prob), 1e-6) == 1

    # extract image embeddings.
    embedding = predictor.extract_embedding({"123": [dog_image_name, cat_image_name]})
    assert list(embedding.keys()) == ["123"]
    for v in embedding.values():
        assert v.shape == (2, 512) or v.shape == (2, 768)

    # extract text embeddings.
    embedding = predictor.extract_embedding({"xyz": [bird_text, dog_text, cat_text]})
    assert list(embedding.keys()) == ["xyz"]
    for v in embedding.values():
        assert v.shape == (3, 512) or v.shape == (3, 768)

    # extract embeddings for both images and texts.
    embedding = predictor.extract_embedding({"image": [cat_image_name, dog_image_name], "text": [bird_text, dog_text]})
    assert list(embedding.keys()).sort() == ["image", "text"].sort()
    for v in embedding.values():
        assert v.shape == (2, 512) or v.shape == (2, 768)

    # invalid API usage 1: passing one dictionary, but different keys have inconsistent list lengths.
    with pytest.raises(ValueError):
        pred = predictor.predict({"image": [cat_image_name, dog_image_name], "text": [cat_text]})

    with pytest.raises(ValueError):
        embedding = predictor.extract_embedding({"image": [cat_image_name], "text": [cat_text, bird_text]})


@pytest.mark.parametrize(
    "checkpoint_name,num_gpus",
    [
        ("swin_tiny_patch4_window7_224", -1),
        ("vit_tiny_patch16_224", -1),
        ("resnet18", -1),
        ("legacy_seresnet18", -1),
    ],
)
def test_timm_zero_shot(checkpoint_name, num_gpus):
    cat_image_name, dog_image_name = download_sample_images()

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.names": ["timm_image"],
            "model.timm_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        problem_type="zero_shot_image_classification",
    )

    pred = predictor.predict({"image": [cat_image_name, dog_image_name]})
    assert pred.shape == (2,)

    prob = predictor.predict_proba({"image": [cat_image_name, dog_image_name]})
    assert prob.shape == (2, 1000)

    features = predictor.extract_embedding({"abc": [cat_image_name, dog_image_name]})
    assert features["abc"].ndim == 2 and features["abc"].shape[0] == 2

    features, masks = predictor.extract_embedding({"abc": [cat_image_name, dog_image_name]}, return_masks=True)
    assert features["abc"].ndim == 2 and features["abc"].shape[0] == 2
    assert np.all(masks["abc"] == np.array([1, 1]))

    features, masks = predictor.extract_embedding(
        {"abc": [cat_image_name], "123": [dog_image_name]}, return_masks=True
    )
    assert features["abc"].ndim == 2 and features["abc"].shape[0] == 1
    assert features["123"].ndim == 2 and features["123"].shape[0] == 1
    assert np.all(masks["abc"] == np.array([1]))
    assert np.all(masks["123"] == np.array([1]))


def test_matcher_text_similarity():
    matcher = MultiModalPredictor(
        problem_type="text_similarity",
        hyperparameters={"model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2"},
    )
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ]
    queries = [
        "A man is eating pasta.",
        "Someone in a gorilla costume is playing a set of drums.",
        "A cheetah chases prey on across a field.",
    ]
    query_embeddings = matcher.extract_embedding(queries)
    corpus_embeddings = matcher.extract_embedding(corpus)
    assert len(query_embeddings) == len(queries)
    assert len(corpus_embeddings) == len(corpus)

    query_embeddings = matcher.extract_embedding(queries, signature="query")
    corpus_embeddings = matcher.extract_embedding(corpus, signature="response")
    assert len(query_embeddings) == len(queries)
    assert len(corpus_embeddings) == len(corpus)

    query_embeddings = matcher.extract_embedding({"abc": queries})
    corpus_embeddings = matcher.extract_embedding({"abc": corpus})
    assert len(query_embeddings) == len(queries)
    assert len(corpus_embeddings) == len(corpus)

    query_embeddings = matcher.extract_embedding({"abc": queries}, signature="query")
    corpus_embeddings = matcher.extract_embedding({"abc": corpus}, signature="response")
    assert len(query_embeddings) == len(queries)
    assert len(corpus_embeddings) == len(corpus)
