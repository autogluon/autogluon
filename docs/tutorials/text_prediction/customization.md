# Text Prediction - Customized Hyperparameter Search
:label:`sec_textprediction_customization`

This tutorial teaches you how to control the hyperparameter tuning process in `TextPrediction` by specifying:

- A custom search space of candidate hyperparameter values to consider.
- Which hyperparameter optimization algorithm should be used to actually search through this space.

```{.python .input}
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Paraphrase Identification

We consider a Paraphrase Identification task for illustration. Given a pair of sentences, the goal is to predict whether or not one sentence is a restatement of the other (a binary classification task). Here we train models on the [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398) dataset.
For quick demonstration, we will subsample the training data and only use 800 samples.

```{.python .input}
from autogluon.core.utils.loaders import load_pd

train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/mrpc/train.parquet')
dev_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/mrpc/dev.parquet')
rand_idx = np.random.permutation(np.arange(len(train_data)))[:800]
train_data = train_data.iloc[rand_idx]
train_data.reset_index(inplace=True, drop=True)
train_data.head(10)
```


```{.python .input}
from autogluon_contrib_nlp.data.tokenizers import MosesTokenizer
tokenizer = MosesTokenizer('en')  # just used to display sentences
row_index = 2
print('Paraphrase example:')
print('Sentence1: ', tokenizer.decode(train_data['sentence1'][row_index].split()))
print('Sentence2: ', tokenizer.decode(train_data['sentence2'][row_index].split()))
print('Label: ', train_data['label'][row_index])

row_index = 3
print('\nNot Paraphrase example:')
print('Sentence1:', tokenizer.decode(train_data['sentence1'][row_index].split()))
print('Sentence2:', tokenizer.decode(train_data['sentence2'][row_index].split()))
print('Label:', train_data['label'][row_index])
```

## Perform HPO over a Customized Search Space with Random Search

To control which hyperparameter values are considered during `fit()`, we specify the `hyperparameters` argument.
Rather than specifying a particular fixed value for a hyperparameter, we can specify a space of values to search over via `ag.space`.
We can also specify which HPO algorithm to use for the search via `search_strategy` (a simple [random search](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) is specified below).
In this example, we search for good values of the following hyperparameters:

- warmup
- learning rate
- dropout before the first task-specific layer
- layer-wise learning rate decay
- number of task-specific layers

```{.python .input}
import autogluon.core as ag
from autogluon.text import TextPrediction as task

hyperparameters = {
    'models': {
            'BertForTextPredictionBasic': {
                'search_space': {
                    'model.network.agg_net.mid_units': ag.space.Int(32, 128),
                    'model.network.agg_net.data_dropout': ag.space.Categorical(False, True),
                    'optimization.num_train_epochs': 4,
                    'optimization.warmup_portion': ag.space.Real(0.1, 0.2),
                    'optimization.layerwise_lr_decay': ag.space.Real(0.8, 1.0),
                    'optimization.lr': ag.space.Real(1E-5, 1E-4)
                }
            },
    },
    'hpo_params': {
        'search_strategy': 'random'  # perform HPO via simple random search
    }
}
```

We can now call `fit()` with hyperparameter-tuning over our custom search space. 
Below `num_trials` controls the maximal number of different hyperparameter configurations for which AutoGluon will train models (5 models are trained under different hyperparameter configurations in this case). To achieve good performance in your applications, you should use larger values of `num_trials`, which may identify superior hyperparameter values but will require longer runtimes.

```{.python .input}
predictor_mrpc = task.fit(train_data,
                          label='label',
                          hyperparameters=hyperparameters,
                          num_trials=2,  # increase this to achieve good performance in your applications
                          time_limits=60 * 2,
                          ngpus_per_trial=1,
                          seed=123,
                          output_directory='./ag_mrpc_random_search')
```

We can again evaluate our model's performance on separate test data.

```{.python .input}
dev_score = predictor_mrpc.evaluate(dev_data, metrics=['acc', 'f1'])
print('Best Config = {}'.format(predictor_mrpc.results['best_config']))
print('Total Time = {}s'.format(predictor_mrpc.results['total_time']))
print('Accuracy = {:.2f}%'.format(dev_score['acc'] * 100))
print('F1 = {:.2f}%'.format(dev_score['f1'] * 100))
```

And also use the model to predict whether new sentence pairs are paraphrases of each other or not.

```{.python .input}
sentence1 = 'It is simple to solve NLP problems with AutoGluon.'
sentence2 = 'With AutoGluon, it is easy to solve NLP problems.'
sentence3 = 'AutoGluon gives you a very bad user experience for solving NLP problems.'
prediction1 = predictor_mrpc.predict({'sentence1': [sentence1], 'sentence2': [sentence2]})
prediction1_prob = predictor_mrpc.predict_proba({'sentence1': [sentence1], 'sentence2': [sentence2]})
print('A = "{}"'.format(sentence1))
print('B = "{}"'.format(sentence2))
print('Prediction = "{}"'.format(prediction1[0] == 1))
print('Prob = "{}"'.format(prediction1_prob[0]))
print('')
prediction2 = predictor_mrpc.predict({'sentence1': [sentence1], 'sentence2': [sentence3]})
prediction2_prob = predictor_mrpc.predict_proba({'sentence1': [sentence1], 'sentence2': [sentence3]})
print('A = "{}"'.format(sentence1))
print('B = "{}"'.format(sentence3))
print('Prediction = "{}"'.format(prediction2[0] == 1))
print('Prob = "{}"'.format(prediction2_prob[0]))
```

## Use Bayesian Optimization

Instead of random search, we can perform HPO via [Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/).
Here we specify **bayesopt** as the searcher.


```{.python .input}
hyperparameters['hpo_params'] = {
    'search_strategy': 'bayesopt'
}

predictor_mrpc_bo = task.fit(train_data, label='label',
                                hyperparameters=hyperparameters,
                                time_limits=60 * 2,
                                num_trials=2,  # increase this to get good performance in your applications
                                ngpus_per_trial=1, seed=123,
                                output_directory='./ag_mrpc_custom_space_fifo_bo')
```


```{.python .input}
dev_score = predictor_mrpc_bo.evaluate(dev_data, metrics=['acc', 'f1'])
print('Best Config = {}'.format(predictor_mrpc_bo.results['best_config']))
print('Total Time = {}s'.format(predictor_mrpc_bo.results['total_time']))
print('Accuracy = {:.2f}%'.format(dev_score['acc'] * 100))
print('F1 = {:.2f}%'.format(dev_score['f1'] * 100))
```


```{.python .input}
predictions = predictor_mrpc_bo.predict(dev_data)
prediction1 = predictor_mrpc_bo.predict({'sentence1': [sentence1], 'sentence2': [sentence2]})
prediction1_prob = predictor_mrpc_bo.predict_proba({'sentence1': [sentence1], 'sentence2': [sentence2]})
print('A = "{}"'.format(sentence1))
print('B = "{}"'.format(sentence2))
print('Prediction = "{}"'.format(prediction1[0] == 1))
print('Prob = "{}"'.format(prediction1_prob[0]))
print('')
prediction2 = predictor_mrpc_bo.predict({'sentence1': [sentence1], 'sentence2': [sentence3]})
prediction2_prob = predictor_mrpc_bo.predict_proba({'sentence1': [sentence1], 'sentence2': [sentence3]})
print('A = "{}"'.format(sentence1))
print('B = "{}"'.format(sentence3))
print('Prediction = "{}"'.format(prediction2[0] == 1))
print('Prob = "{}"'.format(prediction2_prob[0]))
```


## Use Hyperband

Alternatively, we can instead use the [Hyperband algorithm](https://arxiv.org/pdf/1603.06560.pdf) for HPO.
Hyperband will try multiple hyperparameter configurations simultaneously and will early stop training under poor configurations to free compute resources for exploring new hyperparameter configurations. It may be able to identify good hyperparameter values more quickly than other search strategies in your applications.


```{.python .input}
scheduler_options = {'max_t': 40}  # Maximal number of epochs for training the neural network
hyperparameters['hpo_params'] = {
    'search_strategy': 'hyperband',
    'scheduler_options': scheduler_options
}
```


```{.python .input}
predictor_mrpc_hyperband = task.fit(train_data, label='label',
                                    hyperparameters=hyperparameters,
                                    time_limits=60 * 2, ngpus_per_trial=1, seed=123,
                                    output_directory='./ag_mrpc_custom_space_hyperband')
```


```{.python .input}
dev_score = predictor_mrpc_hyperband.evaluate(dev_data, metrics=['acc', 'f1'])
print('Best Config = {}'.format(predictor_mrpc_hyperband.results['best_config']))
print('Total Time = {}s'.format(predictor_mrpc_hyperband.results['total_time']))
print('Accuracy = {:.2f}%'.format(dev_score['acc'] * 100))
print('F1 = {:.2f}%'.format(dev_score['f1'] * 100))
```


```{.python .input}
predictions = predictor_mrpc_hyperband.predict(dev_data)
prediction1 = predictor_mrpc_hyperband.predict({'sentence1': [sentence1], 'sentence2': [sentence2]})
prediction1_prob = predictor_mrpc_hyperband.predict_proba({'sentence1': [sentence1], 'sentence2': [sentence2]})
print('A = "{}"'.format(sentence1))
print('B = "{}"'.format(sentence2))
print('Prediction = "{}"'.format(prediction1[0] == 1))
print('Prob = "{}"'.format(prediction1_prob[0]))
print('')
prediction2 = predictor_mrpc_hyperband.predict({'sentence1': [sentence1], 'sentence2': [sentence3]})
prediction2_prob = predictor_mrpc_hyperband.predict_proba({'sentence1': [sentence1], 'sentence2': [sentence3]})
print('A = "{}"'.format(sentence1))
print('B = "{}"'.format(sentence3))
print('Prediction = "{}"'.format(prediction2[0] == 1))
print('Prob = "{}"'.format(prediction2_prob[0]))
```


## Use Hyperband together with Bayesian Optimization

Finally, we can use a combination of [Hyperband and Bayesian Optimization](https://arxiv.org/abs/2003.10865).


```{.python .input}
scheduler_options = {'max_t': 40}
hyperparameters['hpo_params'] = {
    'search_strategy': 'bayesopt_hyperband',
    'scheduler_options': scheduler_options
}
```


```{.python .input}
predictor_mrpc_bohb = task.fit(
    train_data, label='label',
    hyperparameters=hyperparameters,
    time_limits=60 * 2, ngpus_per_trial=1, seed=123,
    output_directory='./ag_mrpc_custom_space_bohb')
```


```{.python .input}
dev_score = predictor_mrpc_bohb.evaluate(dev_data, metrics=['acc', 'f1'])
print('Best Config = {}'.format(predictor_mrpc_bohb.results['best_config']))
print('Total Time = {}s'.format(predictor_mrpc_bohb.results['total_time']))
print('Accuracy = {:.2f}%'.format(dev_score['acc'] * 100))
print('F1 = {:.2f}%'.format(dev_score['f1'] * 100))
```


```{.python .input}
predictions = predictor_mrpc_bohb.predict(dev_data)
prediction1 = predictor_mrpc_bohb.predict({'sentence1': [sentence1], 'sentence2': [sentence2]})
prediction1_prob = predictor_mrpc_bohb.predict_proba({'sentence1': [sentence1], 'sentence2': [sentence2]})
print('A = "{}"'.format(sentence1))
print('B = "{}"'.format(sentence2))
print('Prediction = "{}"'.format(prediction1[0] == 1))
print('Prob = "{}"'.format(prediction1_prob[0]))
print('')
prediction2 = predictor_mrpc_bohb.predict({'sentence1': [sentence1], 'sentence2': [sentence3]})
prediction2_prob = predictor_mrpc_bohb.predict_proba({'sentence1': [sentence1], 'sentence2': [sentence3]})
print('A = "{}"'.format(sentence1))
print('B = "{}"'.format(sentence3))
print('Prediction = "{}"'.format(prediction2[0] == 1))
print('Prob = "{}"'.format(prediction2_prob[0]))
```
