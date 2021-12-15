""" Helper functions related to NN architectures """

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
