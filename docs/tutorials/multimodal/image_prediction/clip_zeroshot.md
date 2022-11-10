# CLIP in AutoMM - Zero-Shot Image Classification

:label:`sec_automm_clip_zeroshot_imgcls`

When you want to classify an image to different classes, it is standard to train an image classifier based on the class names. However, it is tedious to collect training data. And if the collected data is too few or too imbalanced, you may not get a decent image classifier. So you wonder, is there a strong enough model that can handle this situaton without the training efforts?

Actually there is! OpenAI has introduced a model named [CLIP](https://openai.com/blog/clip/), which can be applied to any visual classification benchmark by simply providing the names of the visual categories to be recognized. And its accuracy is high, e.g., CLIP can achieve 76.2% top-1 accuracy on ImageNet without using any of the 1.28M training samples. This performance matches with original supervised ResNet50 on ImageNet, quite promising for a classification task with 1000 classes!

So in this tutorial, let's dive deep into CLIP. We will show you how to use CLIP model to do zero-shot image classification in AutoGluon.

## Simple Demo

Here we provide a simple demo to classify what dog breed is in the picture below.

```{.python .input}
from IPython.display import Image, display
from autogluon.multimodal import download

url = "https://farm4.staticflickr.com/3445/3262471985_ed886bf61a_z.jpg"
dog_image = download(url)

pil_img = Image(filename=dog_image)
display(pil_img)
```

Normally to solve this task, you need to collect some training data (e.g., [the Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)) and train a dog breed classifier. But with CLIP, all you need to do is provide some potential visual categories. CLIP will handle the rest for you.

```{.python .input}
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(problem_type="zero_shot_image_classification")
prob = predictor.predict_proba({"image": [dog_image]}, {"text": ['This is a Husky', 'This is a Golden Retriever', 'This is a German Sheperd', 'This is a Samoyed.']})
print("Label probs:", prob)
```

Clearly, according to the probabilities, we know there is a Husky in the photo (which I think is correct)!

Let's try a harder example. Below is a photo of two Segways. This object class is not common in most existing vision datasets.

```{.python .input}
url = "https://live.staticflickr.com/7236/7114602897_9cf00b2820_b.jpg"
segway_image = download(url)

pil_img = Image(filename=segway_image)
display(pil_img)
```

Given several text queries, CLIP can still predict the segway class correctly with high confidence.

```{.python .input}
prob = predictor.predict_proba({"image": [segway_image]}, {"text": ['segway', 'bicycle', 'wheel', 'car']})
print("Label probs:", prob)
```

This is amazing, right? Now a bit knowledge on why and how CLIP works. CLIP is called Contrastive Language-Image Pre-training. It is trained on a massive number of data (400M image-text pairs). By using a simple loss objective, CLIP tries to predict which out of a set of randomly sampled text is actually paired with an given image in the training dataset. As a result, CLIP models can then be applied to nearly arbitrary visual classification tasks just like the examples we have shown above.

## More about CLIP

CLIP is powerful, and it was designed to mitigate a number of major problems in the standard deep learning approach to computer vision, such as costly datasets, closed set prediction and poor generalization performance. CLIP is a good solution to many problems, however, it is not the ultimate solution. CLIP has its own limitations. For example, CLIP is vulnerable to typographic attacks, i.e., if you add some text to an image, CLIP's predictions will be easily affected by the text. Let's see one example from OpenAI's blog post on [multimodal neurons](https://openai.com/blog/multimodal-neurons/).

Suppose we have a photo of a Granny Smith apple,

```{.python .input}
url = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-blank.jpg"
image_path = download(url)

pil_img = Image(filename=image_path)
display(pil_img)
```

We then try to classify this image to several classes, such as Granny Smith, iPod, library, pizza, toaster and dough.

```{.python .input}
prob = predictor.predict_proba({"image": [image_path]}, {"text": ['Granny Smith', 'iPod', 'library', 'pizza', 'toaster', 'dough']})
print("Label probs:", prob)
```

We can see that zero-shot classification works great, it predicts apple with almost 100% confidence. But if we add a text to the apple like this,

```{.python .input}
url = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-ipod.jpg"
image_path = download(url)

pil_img = Image(filename=image_path)
display(pil_img)
```

Then we use the same class names to perform zero-shot classification,

```{.python .input}
prob = predictor.predict_proba({"image": [image_path]}, {"text": ['Granny Smith', 'iPod', 'library', 'pizza', 'toaster', 'dough']})
print("Label probs:", prob)
```

Suddenly, the apple becomes iPod.

CLIP also has other limitations. If you are interested, you can read [CLIP paper](https://arxiv.org/abs/2103.00020) for more details. Or you can stay here, play with your own examples!

## Other Examples

You may go to [AutoMM Examples](https://github.com/awslabs/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization

To learn how to customize AutoMM, please refer to :ref:`sec_automm_customization`.
