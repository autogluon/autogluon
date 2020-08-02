# Text Prediction - Customization and HPO
:label:`sec_text_prediction_customization`

In this tutorial, we will learn how to customize the `TextPrediction` task in AutoGluon. We will use the "Paraphrasing Identification" for illustration. The goal of the task is to identify whether one sentence is a restatement of the other. In this example, we will use the [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398). 


```{.python .input}
import numpy as np
import warnings
warnings.filterwarnings('ignore')
np.random.seed(123)
```

## Load Data


```{.python .input}
from autogluon.utils.tabular.utils.loaders import load_pd

train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/mrpc/train.parquet')
dev_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/glue/mrpc/dev.parquet')
train_data.head(10)
```


```{.python .input}
from autogluon_contrib_nlp.data.tokenizers import MosesTokenizer
tokenizer = MosesTokenizer('en')
print('Paraphrase:')
print('Sentence1: ', tokenizer.decode(train_data['sentence1'][5].split()))
print('Sentence2: ', tokenizer.decode(train_data['sentence2'][5].split()))
print('Label: ', train_data['label'][2])

print('\nNot Paraphrase:')
print('Sentence1:', tokenizer.decode(train_data['sentence1'][8].split()))
print('Sentence2:', tokenizer.decode(train_data['sentence2'][8].split()))
print('Label:', train_data['label'][3])
```

## Explore a Custoimzed Search Space with Random Search

Here, we can set the `hyperparameters` argument and specify the search space with `ag.space`. In this example, we search for the `warmup_portion`, `learning_rate`, `data_dropout`, `layerwise_lr_decay`, and `num_layers`.


```{.python .input}
import autogluon as ag
from autogluon import TextPrediction as task

hyperparameters = {
    'models': {
            'BertForTextPredictionBasic': {
                'search_space': {
                    'model.network.agg_net.num_layers': ag.space.Categorical(0, 1, 2),
                    'model.network.agg_net.data_dropout': ag.space.Categorical(False, True),
                    'optimization.num_train_epochs': 4,
                    'optimization.warmup_portion': ag.space.Real(0.1, 0.2),
                    'optimization.layerwise_lr_decay': ag.space.Real(0.8, 1.0),
                    'optimization.lr': ag.space.Real(1E-5, 1E-4)
                }
            }
    },
    'hpo_params': {
        'scheduler': 'fifo',
        'search_strategy': 'random'
    }
}

predictor_mrpc = task.fit(train_data, label='label',
                          hyperparameters=hyperparameters,
                          num_trials=5,
                          time_limits=60 * 5, ngpus_per_trial=1, seed=123,
                          output_directory='./ag_mrpc_random_search')
```


```{.python .input}
dev_score = predictor_mrpc.evaluate(dev_data, metrics=['acc', 'f1'])
print('Best Config = {}'.format(predictor_mrpc.results['best_config']))
print('Total Time = {}s'.format(predictor_mrpc.results['total_time']))
print('Accuracy = {:.2f}'.format(dev_score['acc']))
print('F1 = {:.2f}'.format(dev_score['f1']))
```


```{.python .input}
predictions = predictor_mrpc.predict(dev_data)
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


```{.python .input}
## Use Bayesian Optimization
```


```{.python .input}
hyperparameters['hpo_params'] = {
    'scheduler': 'fifo',
    'search_strategy': 'skopt'
}

predictor_mrpc_skopt = task.fit(train_data, label='label',
                                hyperparameters=hyperparameters,
                                time_limits=60 * 5,
                                num_trials=5,
                                ngpus_per_trial=1, seed=123,
                                output_directory='./ag_mrpc_custom_space_fifo_skopt')
```


```{.python .input}
dev_score = predictor_mrpc_skopt.evaluate(dev_data, metrics=['acc', 'f1'])
print('Best Config = {}'.format(predictor_mrpc_skopt.results['best_config']))
print('Total Time = {}s'.format(predictor_mrpc_skopt.results['total_time']))
print('Accuracy = {:.2f}'.format(dev_score['acc']))
print('F1 = {:.2f}'.format(dev_score['f1']))
```


```{.python .input}
predictions = predictor_mrpc_skopt.predict(dev_data)
prediction1 = predictor_mrpc_skopt.predict({'sentence1': [sentence1], 'sentence2': [sentence2]})
prediction1_prob = predictor_mrpc_skopt.predict_proba({'sentence1': [sentence1], 'sentence2': [sentence2]})
print('A = "{}"'.format(sentence1))
print('B = "{}"'.format(sentence2))
print('Prediction = "{}"'.format(prediction1[0] == 1))
print('Prob = "{}"'.format(prediction1_prob[0]))
print('')
prediction2 = predictor_mrpc_skopt.predict({'sentence1': [sentence1], 'sentence2': [sentence3]})
prediction2_prob = predictor_mrpc_skopt.predict_proba({'sentence1': [sentence1], 'sentence2': [sentence3]})
print('A = "{}"'.format(sentence1))
print('B = "{}"'.format(sentence3))
print('Prediction = "{}"'.format(prediction2[0] == 1))
print('Prob = "{}"'.format(prediction2_prob[0]))
```

## Use Hyperband


```{.python .input}
hyperparameters['hpo_params'] = {
    'scheduler': 'hyperband',
    'search_strategy': 'random',
    'max_t': 40,
}
```


```{.python .input}
predictor_mrpc_hyperband = task.fit(train_data, label='label',
                                    hyperparameters=hyperparameters,
                                    time_limits=60 * 5, ngpus_per_trial=1, seed=123,
                                    output_directory='./ag_mrpc_custom_space_hyperband')
```


```{.python .input}
dev_score = predictor_mrpc_hyperband.evaluate(dev_data, metrics=['acc', 'f1'])
print('Best Config = {}'.format(predictor_mrpc_hyperband.results['best_config']))
print('Total Time = {}s'.format(predictor_mrpc_hyperband.results['total_time']))
print('Accuracy = {:.2f}'.format(dev_score['acc']))
print('F1 = {:.2f}'.format(dev_score['f1']))
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
