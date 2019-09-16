from fastai.text import *
from sklearn.decomposition import PCA


def plot_accuracy_board(val_to_pub):
    plt.figure()
    # ax = plt.axes(xlim=(0, 1), ylim=(0, 1), title='Public accuracy vs validation accuracy', xlabel='val_acc', ylabel='pub_acc')
    ax = plt.axes(xlim=(0.3, 0.6), ylim=(0.3, 0.6), title='Public accuracy vs validation accuracy', xlabel='val_acc', ylabel='pub_acc')
    ax.scatter(val_to_pub[:, 0], val_to_pub[:, 1])
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, linestyle='dashed', linewidth=0.5)
    for i, txt in enumerate(np.arange(1, val_to_pub.shape[0] + 1)):
        ax.annotate(txt, (val_to_pub[:, 0][i] - 0.01, val_to_pub[:, 1][i] + 0.03))


def get_embed_vectors(cat, embeds, emb_cat_names):
    cat2i = {c: i for i, c in enumerate(emb_cat_names)}
    cat2emb = {i: [c, embeds[i].num_embeddings, embeds[i].embedding_dim] for i, c in enumerate(emb_cat_names)};
    i = cat2i[cat]
    c, num, dim = cat2emb[i]
    vecs = to_np(embeds[i](LongTensor(range(num))))
    return {'name': c, 'num_embeddings': num, 'embedding_dim': dim, 'vecs': vecs}


def plot_vectors_pcs(cats_to_display, learn, data, model_cat_names, **kwargs):
    embeds = learn.model.embeds
    for i, cat in enumerate(cats_to_display):
        vecs = get_embed_vectors(cat, embeds, model_cat_names)
        num = vecs['num_embeddings']
        dim = vecs['embedding_dim']
        vecs = vecs['vecs']
        if dim > 2:
            pca = PCA(n_components=2)
            cat_pca = pca.fit(vecs.T).components_
        else:
            cat_pca = vecs.T
        X = cat_pca[0]
        Y = cat_pca[1]
        f = plt.figure(**kwargs)
        plt.subplot(len(cats_to_display) // 3 + 1, 3, i + 1)
        plt.subplots_adjust(hspace=0.4)
        plt.title(f'{cat}({num}->{dim})')
        plt.scatter(X, Y)
        for i, x, y in zip(range(num), X, Y):
            plt.text(x, y, str(data.x.classes[cat][i])[:16], color=np.random.rand(3) * 0.7, fontsize=11)


def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
