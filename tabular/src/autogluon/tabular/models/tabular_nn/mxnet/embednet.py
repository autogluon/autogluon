import numpy as np
import mxnet as mx
from mxnet import nd, gluon


class NumericBlock(gluon.HybridBlock):
    """ Single Dense layer that jointly embeds all numeric and one-hot features """
    def __init__(self, params, **kwargs):
        super(NumericBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.body = gluon.nn.Dense(params['numeric_embed_dim'], activation=params['activation'])
            
    def hybrid_forward(self, F, x):
        return self.body(x)


class EmbedBlock(gluon.HybridBlock):
    """ Used to embed a single embedding feature. """
    def __init__(self, embed_dim, num_categories, **kwargs):
        super(EmbedBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.body = gluon.nn.Embedding(input_dim=num_categories, output_dim=embed_dim,
                                           weight_initializer=mx.init.Orthogonal(scale=0.1, rand_type='uniform')) # for Xavier-style: scale = np.sqrt(3/float(embed_dim))
    
    def hybrid_forward(self, F, x):
        return self.body(x)


class FeedforwardBlock(gluon.HybridBlock):
    """ Standard Feedforward layers """
    def __init__(self, params, num_net_outputs, **kwargs):
        super(FeedforwardBlock, self).__init__(**kwargs)
        layers = params['layers']
        with self.name_scope():
            self.body = gluon.nn.HybridSequential()
            if params['use_batchnorm']:
                 self.body.add(gluon.nn.BatchNorm())
            if params['dropout_prob'] > 0:
                self.body.add(gluon.nn.Dropout(params['dropout_prob']))
            for i in range(len(layers)):
                layer_width = layers[i]
                if layer_width < 1 or int(layer_width) != layer_width:
                    raise ValueError("layers must be ints >= 1")
                self.body.add(gluon.nn.Dense(layer_width, activation=params['activation']))
                if params['use_batchnorm']:
                    self.body.add(gluon.nn.BatchNorm())
                if params['dropout_prob'] > 0:
                    self.body.add(gluon.nn.Dropout(params['dropout_prob']))
            self.body.add(gluon.nn.Dense(num_net_outputs, activation=None))
    
    def hybrid_forward(self, F, x):
        return self.body(x)


class WideAndDeepBlock(gluon.HybridBlock):
    """ Standard feedforward layers with a single skip connection from output directly to input (ie. deep and wide network).
    """
    def __init__(self, params, num_net_outputs, **kwargs):
        super(WideAndDeepBlock, self).__init__(**kwargs)
        self.deep = FeedforwardBlock(params, num_net_outputs, **kwargs)
        with self.name_scope(): # Skip connection, ie. wide network branch
            self.wide = gluon.nn.Dense(num_net_outputs, activation=None)
    
    def hybrid_forward(self, F, x):
        return self.deep(x) + self.wide(x)


class EmbedNet(gluon.Block): # TODO: hybridize?
    """ Gluon net with input layers to handle numerical data & categorical embeddings
        which are concatenated together after input layer and then passed into feedforward network.
        If architecture_desc != None, then we assume EmbedNet has already been previously created,
        and we create a new EmbedNet based on the provided architecture description 
        (thus ignoring train_dataset, params, num_net_outputs). 
    """
    def __init__(self, train_dataset=None, params=None, num_net_outputs=None, architecture_desc=None, ctx=None, **kwargs):
        if (architecture_desc is None) and (train_dataset is None or params is None or num_net_outputs is None):
            raise ValueError("train_dataset, params, num_net_outputs cannot = None if architecture_desc=None")
        super(EmbedNet, self).__init__(**kwargs)
        if architecture_desc is None: # Adpatively specify network architecture based on training dataset
            self.from_logits = False
            self.has_vector_features = train_dataset.has_vector_features()
            self.has_embed_features = train_dataset.num_embed_features() > 0
            self.has_language_features = train_dataset.num_language_features() > 0
            if self.has_embed_features:
                num_categs_per_feature = train_dataset.getNumCategoriesEmbeddings()
                embed_dims = getEmbedSizes(train_dataset, params, num_categs_per_feature)
        else: # Ignore train_dataset, params, etc. Recreate architecture based on description:
            self.architecture_desc = architecture_desc
            self.has_vector_features = architecture_desc['has_vector_features']
            self.has_embed_features = architecture_desc['has_embed_features']
            self.has_language_features = architecture_desc['has_language_features']
            self.from_logits = architecture_desc['from_logits']
            num_net_outputs = architecture_desc['num_net_outputs']
            params = architecture_desc['params']
            if self.has_embed_features:
                num_categs_per_feature = architecture_desc['num_categs_per_feature']
                embed_dims = architecture_desc['embed_dims']
        
        # Define neural net parameters:
        if self.has_vector_features:
            self.numeric_block = NumericBlock(params)
        if self.has_embed_features:
            self.embed_blocks = gluon.nn.HybridSequential()
            for i in range(len(num_categs_per_feature)):
                self.embed_blocks.add(EmbedBlock(embed_dims[i], num_categs_per_feature[i]))
        if self.has_language_features:
            self.text_block = None
            raise NotImplementedError("text data cannot be handled")
        if params['network_type'] == 'feedforward':
            self.output_block = FeedforwardBlock(params, num_net_outputs)
        elif params['network_type'] == 'widedeep':
            self.output_block = WideAndDeepBlock(params, num_net_outputs)
        else:
            raise ValueError("unknown network_type specified: %s" % params['network_type'])
        
        y_range = params['y_range'] # Used specifically for regression. = None for classification.
        self.y_constraint = None # determines if Y-predictions should be constrained
        if y_range is not None:
            if y_range[0] == -np.inf and y_range[1] == np.inf:
                self.y_constraint = None # do not worry about Y-range in this case
            elif y_range[0] >= 0 and y_range[1] == np.inf:
                self.y_constraint = 'nonnegative'
            elif y_range[0] == -np.inf and y_range[1] <= 0:
                self.y_constraint = 'nonpositive'
            else:
                self.y_constraint = 'bounded'
            self.y_lower = nd.array(params['y_range'][0]).reshape(1,)
            self.y_upper = nd.array(params['y_range'][1]).reshape(1,)
            if ctx is not None:
                self.y_lower = self.y_lower.as_in_context(ctx)
                self.y_upper = self.y_upper.as_in_context(ctx)
            self.y_span = self.y_upper - self.y_lower
        
        if architecture_desc is None: # Save Architecture description
            self.architecture_desc = {'has_vector_features': self.has_vector_features, 
                                  'has_embed_features': self.has_embed_features,
                                  'has_language_features': self.has_language_features,
                                  'params': params, 'num_net_outputs': num_net_outputs,
                                  'from_logits': self.from_logits}
            if self.has_embed_features:
                self.architecture_desc['num_categs_per_feature'] = num_categs_per_feature
                self.architecture_desc['embed_dims'] = embed_dims
            if self.has_language_features:
                self.architecture_desc['text_TODO'] = None # TODO: store text architecture
    
    def forward(self, data_batch):
        if self.has_vector_features:
            numerical_data = data_batch['vector'] # NDArray
            numerical_activations = self.numeric_block(numerical_data)
            input_activations = numerical_activations
        if self.has_embed_features:
            embed_data = data_batch['embed'] # List

            # TODO: Remove below lines or write logic to switch between using these lines and the multithreaded version once multithreaded version is optimized
            embed_activations = self.embed_blocks[0](embed_data[0])
            for i in range(1, len(self.embed_blocks)):
                embed_activations = nd.concat(embed_activations,
                                              self.embed_blocks[i](embed_data[i]), dim=2)

            # TODO: Optimize below to perform better before using
            # lock = threading.Lock()
            # results = {}
            #
            # def _worker(i, results, embed_block, embed_data, is_recording, is_training, lock):
            #     if is_recording:
            #         with mx.autograd.record(is_training):
            #             output = embed_block(embed_data)
            #     else:
            #         output = embed_block(embed_data)
            #     output.wait_to_read()
            #     with lock:
            #         results[i] = output
            #
            # is_training = mx.autograd.is_training()
            # is_recording = mx.autograd.is_recording()
            # threads = [threading.Thread(target=_worker,
            #                     args=(i, results, embed_block, embed_data,
            #                           is_recording, is_training, lock),
            #                     )
            #    for i, (embed_block, embed_data) in
            #    enumerate(zip(self.embed_blocks, embed_data))]
            #
            # for thread in threads:
            #     thread.start()
            # for thread in threads:
            #     thread.join()
            #
            # embed_activations = []
            # for i in range(len(results)):
            #     output = results[i]
            #     embed_activations.append(output)
            #
            # #embed_activations = []
            # #for i in range(len(self.embed_blocks)):
            # #    embed_activations.append(self.embed_blocks[i](embed_data[i]))
            # embed_activations = nd.concat(*embed_activations, dim=2)
            embed_activations = embed_activations.flatten()
            if not self.has_vector_features:
                input_activations = embed_activations
            else:
                input_activations = nd.concat(embed_activations, input_activations)
        if self.has_language_features:
            language_data = data_batch['language']
            language_activations = self.text_block(language_data) # TODO: create block to embed text fields
            if (not self.has_vector_features) and (not self.has_embed_features):
                input_activations = language_activations
            else:
                input_activations = nd.concat(language_activations, input_activations)
        if self.y_constraint is None:
            return self.output_block(input_activations)
        else:
            unscaled_pred = self.output_block(input_activations)
            if self.y_constraint == 'nonnegative':
                return self.y_lower + nd.abs(unscaled_pred)
            elif self.y_constraint == 'nonpositive':
                return self.y_upper - nd.abs(unscaled_pred)
            else:
                """
                print("unscaled_pred",unscaled_pred)
                print("nd.sigmoid(unscaled_pred)", nd.sigmoid(unscaled_pred))
                print("self.y_span", self.y_span)
                print("self.y_lower", self.y_lower)
                print("self.y_lower.shape", self.y_lower.shape)
                print("nd.sigmoid(unscaled_pred).shape", nd.sigmoid(unscaled_pred).shape)
                """
                return nd.sigmoid(unscaled_pred) * self.y_span + self.y_lower


""" OLD 
    def _create_embednet_from_architecture(architecture_desc):
        # Recreate network architecture based on provided description
        self.architecture_desc = architecture_desc
        self.has_vector_features = architecture_desc['has_vector_features']
        self.has_embed_features = architecture_desc['has_embed_features']
        self.has_language_features = architecture_desc['has_language_features']
        self.from_logits = architecture_desc['from_logits']
        num_net_outputs = architecture_desc['num_net_outputs']
        params = architecture_desc['params']
        if self.has_vector_features:
            self.numeric_block = NumericBlock(params)
        if self.has_embed_features:
            self.embed_blocks = gluon.nn.HybridSequential()
            num_categs_per_feature = architecture_desc['num_categs_per_feature']
            embed_dims = architecture_desc['embed_dims']
            for i in range(len(num_categs_per_feature)):
                self.embed_blocks.add(EmbedBlock(embed_dims[i], num_categs_per_feature[i]))
        if self.has_language_features:
            self.text_block = architecture_desc['text_TODO']
        if 
        self.output_block = FeedforwardBlock(params, num_net_outputs) # TODO
        self.from_logits = False
"""


def getEmbedSizes(train_dataset, params, num_categs_per_feature):  
    """ Returns list of embedding sizes for each categorical variable.
        Selects this adaptively based on training_datset.
        Note: Assumes there is at least one embed feature.
    """
    max_embedding_dim = params['max_embedding_dim']
    embed_exponent = params['embed_exponent']
    size_factor = params['embedding_size_factor']
    embed_dims = [int(size_factor*max(2, min(max_embedding_dim, 
                                      1.6 * num_categs_per_feature[i]**embed_exponent)))
                   for i in range(len(num_categs_per_feature))]
    return embed_dims
