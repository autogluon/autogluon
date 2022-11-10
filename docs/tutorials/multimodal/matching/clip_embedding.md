# CLIP in AutoMM - Extract Embeddings 
:label:`sec_automm_clip_embedding`

We have shown CLIP's amazing capability in performing zero-shot image classification in our previous tutorial :ref:`sec_automm_clip_zeroshot_imgcls`. Thanks to the contrastive loss objective and trained on millions of image-text pairs, CLIP learns good embeddings for both vision and language, and their connections. Hence, another important use case of CLIP is to extract embeddings for retrieval, matching, ranking kind of tasks.

In this tutorial, we will show you how to use AutoGluon to extract embeddings from CLIP, and then use it for a retrieval problem. 


## Extract Embeddings

CLIP has a two-tower architecture, which means it has two encoders: one for image, the other for text. An overview of CLIP model can be seen in the diagram below. Left shows its pre-training stage, and Right shows its zero-shot predicton stage. By computing the cosine similarity scores between one image embedding and all the text images, we pick the text which has the highest similarity as the prediction.

![CLIP](https://github.com/openai/CLIP/raw/main/CLIP.png)

Given the two encoders, we can extract image embeddings, or text embeddings. And most importantly, embedding extraction can be done offline, only similarity computation needs to be done online. So this means good scalability. 

First, let's download some images. These images are from [COCO datasets](https://cocodataset.org/#home).

```{.python .input}
from autogluon.multimodal import download

urls = ['http://farm4.staticflickr.com/3179/2872917634_f41e6987a8_z.jpg',
        'http://farm4.staticflickr.com/3629/3608371042_75f9618851_z.jpg',
        'https://farm4.staticflickr.com/3795/9591251800_9c9727e178_z.jpg',
        'http://farm8.staticflickr.com/7188/6848765123_252bfca33d_z.jpg',
        'https://farm6.staticflickr.com/5251/5548123650_1a69ce1e34_z.jpg']

image_paths = [download(url) for url in urls]
print(image_paths)
```

Let's extract some image embedding from the CLIP vision encoder,

```{.python .input}
from autogluon.multimodal import MultiModalPredictor
predictor = MultiModalPredictor(problem_type="zero_shot_image_classification")

# extract image embeddings.
image_embeddings = predictor.extract_embedding({"image": image_paths})
print(image_embeddings['image'].shape)  # image (5, 768)
```

The output has a shape of (5, 768), because there are 5 images, each of the image embedding has a dimension of 768. 

Similarly, you can also extract text embeddings from the CLIP text encoder, 

```{.python .input}
# extract text embeddings.
text_embeddings = predictor.extract_embedding({"text": ['There is a carriage in the image']})
print(text_embeddings['text'].shape)  # text (1, 768)
```

Then you can use the embeddings for a range of tasks such as image retrieval, text retrieval, image-text retrieval and matching/ranking. 


## Image retrieval by text query

Suppose we have a large image database (e.g., video footage), now we want to retrieve some images defined by a text query. How can we do this? 

It is simple. First, extract all the image embeddings either online or offline as shown above. Then, compute the text embedding of the text query. Finally, compute the cosine similarity between the text embedding and all the image embeddings and return the top candidates. 

Let's reuse the example we have above. We already have 5 image embeddings in **image_embeddings**, and 1 text embedding in **text_embeddings**, now we normalize the feature and then compute their similarities,

```{.python .input}
image_features = image_embeddings['image']
text_features = text_embeddings['text']

import numpy as np

similarity = np.matmul(image_features, text_features.T)
print(np.argmax(similarity))
```

We can see that we successfully find the image with a carriage in it. 

```{.python .input}
from IPython.display import Image, display
pil_img = Image(filename=image_paths[2])
display(pil_img)
```

If we want to switch to another text query, we simply re-compute text embeddings and do this similarity comparison again,

```{.python .input}
text_embeddings = predictor.extract_embedding({"text": ['There is an airplane over a car.']})
text_features = text_embeddings['text']
similarity = np.matmul(image_features, text_features.T)
print(np.argmax(similarity))
```

Now we find the image most corresponding to the text query. 

```{.python .input}
pil_img = Image(filename=image_paths[4])
display(pil_img)
```


## Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.


## Customization

To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
