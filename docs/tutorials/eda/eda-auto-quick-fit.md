# Automated Quick Model Fit

The purpose of this feature is to provide a quick and easy way to obtain a preliminary understanding of the
relationships between the target variable and the independent variables in a dataset.

This functionality automatically splits the training data, fits a simple regression or classification model to the
data and generates insights: model performance metrics, feature importance and prediction result insights.

To inspect the prediction quality, a confusion matrix is displayed for classification problems and scatter plot for
regression problems. Both representation allow the user to see the difference between actual and predicted values.

The insights highlight two subsets of the model predictions:

- predictions with the largest classification error. Rows listed in this section are candidates for inspecting why the
  model made the mistakes
- predictions with the least distance from the other class. Rows in this category are most 'undecided'. They are useful
  as an examples of data which is close to a decision boundary between the classes. The model would benefit from having
  more data for similar cases.

## Classification Example

We will start with getting titanic dataset and performing a quick one-line overview to get the information.

```{.python .input}
import pandas as pd
import autogluon.eda.auto as auto

df_train = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/titanic/train.csv')
df_test = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/titanic/test.csv')
target_col = 'Survived'

auto.quick_fit(df_train, target_col, show_feature_importance_barplots=True)
```

## Regression Example

In the previous section we tried a classification example. Let's try a regression. It has a few differences.
We are also going to store the fitted model by specifying `return_state` and `save_model_to_state` parameters.
This will allow us to use the model to predict test values later.

It is a large dataset, so we'll keep only a few columns for this tutorial.

```{.python .input}
df_train = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/AmesHousingPriceRegression/train_data.csv')
df_test = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/AmesHousingPriceRegression/test_data.csv')
target_col = 'SalePrice'

keep_cols = [
  'Overall.Qual', 'Gr.Liv.Area', 'Neighborhood', 'Total.Bsmt.SF', 'BsmtFin.SF.1',
  'X1st.Flr.SF', 'Bsmt.Qual', 'Garage.Cars', 'Half.Bath', 'Year.Remod.Add', target_col
]

df_train = df_train[[c for c in df_train.columns if c in keep_cols]][:500]
df_test = df_test[[c for c in df_test.columns if c in keep_cols]][:500]


state = auto.quick_fit(df_train, target_col, return_state=True, save_model_to_state=True)
```

## Using a fitted model

Now let's get the `model` from `state`, perform the prediction on `df_test` and quickly visualize the results using 
`auto.analyze_interaction()` tool:   

```{.python .input}
model = state.model
y_pred = model.predict(df_test)
auto.analyze_interaction(
    train_data=pd.DataFrame({'SalePrice_Pred': y_pred}), 
    x='SalePrice_Pred', 
    fit_distributions=['johnsonsu', 'norm', 'exponnorm']
)
```
