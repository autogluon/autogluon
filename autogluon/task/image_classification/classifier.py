import os
import pickle
import mxnet as mx
import matplotlib.pyplot as plt
from mxnet.gluon.data.vision import transforms

from .utils import *
from .metrics import get_metric_instance
from ..base.base_predictor import BasePredictor
from ...utils import in_ipynb, save, load
if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

class Classifier(BasePredictor):
    """
    Classifier returned by task.fit()

    Example user workflow:
    """
    def __init__(self, model, results, eval_func, scheduler_checkpoint,
                 args, **kwargs):
        self.model = model
        self.eval_func = eval_func
        self.results = results
        self.scheduler_checkpoint = scheduler_checkpoint
        self.args = args

    @classmethod
    def load(cls, checkpoint):
        state_dict = load(checkpoint)
        args = state_dict['args']
        results = state_dict['results']
        eval_func = pickle.loads(state_dict['eval_func'])
        scheduler_checkpoint = state_dict['scheduler_checkpoint']
        model_params = state_dict['model_params']

        model_args = copy.deepcopy(args)
        model_args.update(results['best_config'])
        model = get_network(args.net, args.dataset.num_classes)
        update_params(model, model_params)
        return cls(eval_func, model, eval_func, scheduler_checkpoint, args)

    def state_dict(self, destination=None):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination['model_params'] = collect_params(self.model)
        destination['eval_func'] = pickle.dumps(self.eval_func)
        destination['results'] = self.results
        destination['scheduler_checkpoint'] = self.scheduler_checkpoint
        destination['args'] = self.args
        return destination

    def save(self, checkpoint):
        save(self.state_dict(), checkpoint)

    def predict(self, X, plot=True):
        """ This method should be able to produce predictions regardless if:
            X = single data example (e.g. single image, single document),
            X = batch of many examples, X = task.Dataset object
        """
        """The task predict function given an input.
         Args:
            img: the input
         Example:
            >>> ind, prob = classifier.predict('example.jpg')
        """
        # load and display the image
        img = mx.image.imread(X) if isinstance(X, str) and os.path.isfile(X) else X
        if plot:
            plt.imshow(img.asnumpy())
            plt.show()
        # model inference
        transform_fn = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        img = transform_fn(img)
        proba = self.predict_proba(img)
        ind = mx.nd.argmax(proba, axis=1).astype('int')
        ind = mx.nd.stack(mx.nd.arange(proba.shape[0], ctx=proba.context),
                          ind.astype('float32'))
        return ind, F.gather_nd(proba, ind)

    def predict_proba(self, X):
        """ Produces predicted class probabilities if we are dealing with a classification task.
            In this case, predict() should just be a wrapper around this method to convert predicted probabilties to predicted class labels.
        """
        pred = self.model(img.expand_dims(0))
        return mx.nd.softmax(pred)

    def evaluate(self, dataset, input_size=224, ctx=[mx.cpu()]):
        """The task evaluation function given the test dataset.
         Args:
            dataset: test dataset
         Example:
            >>> from autogluon import ImageClassification as task
            >>> dataset = task.Dataset(name='shopeeiet', test_path='~/data/test')
            >>> test_reward = classifier.evaluate(dataset)
        """
        args = self.args
        net = self.model
        batch_size = args.batch_size * max(len(ctx), 1)
        metric = get_metric_instance(args.metric)
        input_size = net.input_size if hasattr(net, 'input_size') else args.input_size

        _, test_data, batch_fn, _ = get_data_loader(dataset, input_size, batch_size, args.num_workers, False)
        tbar = tqdm(enumerate(test_data))
        for i, batch in tbar:
            self.eval_func(net, batch, batch_fn, metric, ctx)
            _, test_reward = metric.get()
            tbar.set_description('{}: {}'.format(args.metric, test_reward))
        _, test_reward = metric.get()
        return test_reward

    def _save_model(self, *args, **kwargs):
        raise NotImplemented

    def evaluate_predictions(self, y_true, y_pred):
        raise NotImplemented
