# Image-Text Semantic Matching with AutoMM - Zero-Shot
:label:`sec_automm_clip_embedding`

The task of image-text semantic matching refers to measuring the visual-semantic similarity between an image and a sentence. AutoMM supports zero-shot image-text matching by leveraging the powerful [CLIP](https://github.com/openai/CLIP). 
Thanks to the contrastive loss objective and trained on millions of image-text pairs, CLIP learns good embeddings for both vision and language, and their connections. Hence, we can use it to extract embeddings for retrieval and matching.

CLIP has a two-tower architecture, which means it has two encoders: one for image, the other for text. An overview of CLIP model can be seen in the diagram below. Left shows its pre-training stage, and Right shows its zero-shot predicton stage. By computing the cosine similarity scores between one image embedding and all the text images, we pick the text which has the highest similarity as the prediction.

Given the two encoders, we can extract image embeddings, or text embeddings. And most importantly, embedding extraction can be done offline, only similarity computation needs to be done online. So this means good scalability. 
![CLIP](https://github.com/openai/CLIP/raw/main/CLIP.png)


In this tutorial, we will show how the AutoMM's easy-to-use APIs can ship the powerful CLIP to you.

## Prepare Demo Data
First, let's get some texts and download some images. These images are from [COCO datasets](https://cocodataset.org/#home).

```{.python .input}
from autogluon.multimodal import download

texts = [
    "A cheetah chases prey on across a field.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "There is an airplane over a car.",
    "A man is riding a horse.",
    "Two men pushed carts through the woods.",
    "There is a carriage in the image.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
]

urls = ['http://farm4.staticflickr.com/3179/2872917634_f41e6987a8_z.jpg',
        'http://farm4.staticflickr.com/3629/3608371042_75f9618851_z.jpg',
        'https://farm4.staticflickr.com/3795/9591251800_9c9727e178_z.jpg',
        'http://farm8.staticflickr.com/7188/6848765123_252bfca33d_z.jpg',
        'https://farm6.staticflickr.com/5251/5548123650_1a69ce1e34_z.jpg']

image_paths = [download(url) for url in urls]
```


## Extract Embeddings

We need to use `image_text_similarity` as the problem type when initializing the predictor.
```{.python .input}
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(problem_type="image_text_similarity")
```

Let's extract image and text embeddings separately. The image and text data will go through their corresponding encoders, respectively.

```{.python .input}
image_embeddings = predictor.extract_embedding(image_paths, as_tensor=True)
print(image_embeddings.shape)
```

```{.python .input}
text_embeddings = predictor.extract_embedding(texts, as_tensor=True)
print(text_embeddings.shape)
```

Then you can use the embeddings for a range of tasks such as image retrieval and text retrieval. 


## Image Retrieval with Text Query

Suppose we have a large image database (e.g., video footage), now we want to retrieve some images defined by a text query. How can we do this? 

It is simple. First, extract all the image embeddings offline as shown above. Then, extract the text query's embedding. Finally, compute the cosine similarities between the text embedding and all the image embeddings and return the top candidates. 

Suppose we use the text below as the query.
```{.python .input}
print(texts[6])
```

You can directly call our util function `semantic_search` to search semantically similar images.

```{.python .input}
from autogluon.multimodal.utils import semantic_search
hits = semantic_search(
        matcher=predictor,
        query_embeddings=text_embeddings[6][None,],
        response_embeddings=image_embeddings,
        top_k=5,
    )
print(hits)
```

We can see that we successfully find the image with a carriage in it. 

```{.python .input}
from IPython.display import Image, display
pil_img = Image(filename=image_paths[hits[0][0]["response_id"]])
display(pil_img)
```

## Text Retrieval with Image Query

Similarly, given one text database and an image query, we can search texts that match the image. For example, let's search texts for the following image.
```{.python .input}
pil_img = Image(filename=image_paths[4])
display(pil_img)
```

We still use the `semantic_search` function, but switch the assignments of `query_embeddings` and `response_embeddings`.

```{.python .input}
hits = semantic_search(
        matcher=predictor,
        query_embeddings=image_embeddings[4][None,],
        response_embeddings=text_embeddings,
        top_k=5,
    )
print(hits)
```

We can observe that the top-1 text matches the query image.
```{.python .input}
texts[hits[0][0]["response_id"]]
```

## Predict Whether Image-Text Pairs Match
In addition to retrieval, we can let the predictor tell us whether image-text pairs match. 
To do so, we need to initialize the predictor with the additional arguments `query` and `response`, which represent names of image/text and text/image.
```{.python .input}
predictor = MultiModalPredictor(
            query="abc",
            response="xyz",
            problem_type="image_text_similarity",
        )
```

Given image-text pairs, we can make predictions.
```{.python .input}
pred = predictor.predict({"abc": [image_paths[4]], "xyz": [texts[3]]})
print(pred)
```

## Predict Matching Probabilities
It is also easy to predict the matching probabilities. You can make predictions by applying customized thresholds to the probabilities.
```{.python .input}
proba = predictor.predict_proba({"abc": [image_paths[4]], "xyz": [texts[3]]})
print(proba)
```

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.


## Customization

To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
