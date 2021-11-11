# Deploying AutoGluon models with serverless templates
:label:`cloud_aws_lambda_deploy`

After learning how to train a model using AWS SageMaker :ref:`cloud_aws_sagemaker_fit`, in this section we will learn how to deploy 
trained models using AWS Lambda.

## Reducing the model size to minimize AWS Lambda startup times

Because AutoGluon model and containers are larger than a typical Lambda container, it might take some time (60+ seconds) to boot a new Lambda instance from the cold boot.
This could be limiting factor when used with latency-sensitive applications.  To reduce start up times with AWS Lambda it is important to reduce model size to a minimum. 
This can be done by applying  as described in section "Faster presets or hyperparameters" of :ref:`sec_tabularadvanced`:

```{.python}
presets = ['good_quality_faster_inference_only_refit', 'optimize_for_deployment']
```

If the cold boot latency cannot be tolerated, it is recommended to reserve concurrent capacity as described in this article:
[Managing Lambda reserved concurrency](https://docs.aws.amazon.com/lambda/latest/dg/configuration-concurrency.html).

## Creating a base project

To start the project, please follow the following tutorial as a base: 
[Deploying machine learning models with serverless templates](https://aws.amazon.com/blogs/compute/deploying-machine-learning-models-with-serverless-templates/).

To deploy AutoGluon, the following adjustments would be required:

- `Dockerfile` to package AutoGluon runtimes and model files ready for deployment

- Serving `app/app.py` script

When building a docker container it's size can reduce using the following things: 

- use CPU versions of pytorch; if the models to be deployed don't use pytorch, then it is not needed to be installed at all.

- install only the sub-modules required for inference - specifically `autogluon.tabular[all]` will deploy only all tabular models without text and vision modules and their
extra dependencies. This instruction can be further narrowed down to a combination of the following options are: `lightgbm`, `catboost`, `xgboost`, `fastai` and `skex`.

- the trained model is expected to be in `ag_models` directory.

The following `Dockerfile` can be used as a starting point:

```
FROM public.ecr.aws/lambda/python:3.8

RUN yum install libgomp git -y \
 && yum clean all -y && rm -rf /var/cache/yum

ARG TORCH_VER=1.9.1+cpu
ARG TORCH_VISION_VER=0.10.1+cpu
ARG NUMPY_VER=1.19.5
RUN python3.8 -m pip --no-cache-dir install --upgrade --trusted-host pypi.org --trusted-host files.pythonhosted.org pip \
 && python3.8 -m pip --no-cache-dir install --upgrade wheel setuptools \
 && python3.8 -m pip uninstall -y dataclasses \
 && python3.8 -m pip --no-cache-dir install --upgrade torch=="${TORCH_VER}" torchvision=="${TORCH_VISION_VER}" -f https://download.pytorch.org/whl/torch_stable.html \
 && python3.8 -m pip --no-cache-dir install --upgrade numpy==${NUMPY_VER} \
 && python3.8 -m pip --no-cache-dir install --upgrade autogluon.tabular[all]"

COPY app.py ./
COPY ag_models /opt/ml/model/

CMD ["app.lambda_handler"]
```

Application `app/app.py`:

```{.python}
import pandas as pd
from autogluon.tabular import TabularPredictor

model = TabularPredictor.load('/opt/ml/model')
model.persist_models(models='all')


# Lambda handler code
def lambda_handler(event, context):
    df = pd.read_json(event['body'])
    pred_probs = model.predict_proba(df)
    return {
        'statusCode': 200,
        'body': pred_probs.to_json()
    }
```

## Conclusion

In this tutorial we explored how to deploy AutoGluon models using serverless templates. To explore more, refer to the following documentation:

- [Deploying machine learning models with serverless templates](https://aws.amazon.com/blogs/compute/deploying-machine-learning-models-with-serverless-templates/).

- [Managing Lambda reserved concurrency](https://docs.aws.amazon.com/lambda/latest/dg/configuration-concurrency.html)

- [AWS Serverless Application Model (AWS SAM)](https://github.com/aws/serverless-application-model)
