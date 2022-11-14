# Text Semantic Search with AutoMM

### 1. Introduction to semantic embedding

Semantic embedding is one of the main workhorses behind the modern search technology. Instead of directly matching the query to candidates by term frequency (e.g., BM25), a semantic search algorithm matches them by first converting the text $x$ into a feature vector $\phi(x)$ then comparing the similarities using a distance metric defined in that vector space. These feature vectors, known as a "vector embedding", are often trained end-to-end on large text corpus, so that they encode the *semantic* meaning of the text. For example, synonyms are embedded to a similar region of the vector space and relationships between words are often revealed by algebraic operations (see Figure 1 for an example). For these reasons, a vector embedding of text are also known as a **semantic embedding**. With a semantic embedding of the query and the search candidate documents, a search algorithm can often be reduced to finding most similar vectors. This new approach to search is known as **semantic search**.

![Similar sentences have similar embeddings. Image from [Medium](https://medium.com/towards-data-science/fine-grained-analysis-of-sentence-embeddings-a3ff0a42cce5)](https://miro.medium.com/max/1400/0*esMqhzu9WhLiP3bD.jpg)
:width:`500px`

There are three main advantages of using semantic embeddings for a search problem over classical information-retrieval methods (e.g., bag-of-words or TF/IDF).  First, it returns candidates that are related according to the meaning of the text, rather than similar word usage.  This helps to discover paraphrased text and similar concepts described in very different ways. Secondly, semantic search is often more computationally efficient. Vector embeddings of the candidates can be pre-computed and stored in data structures. Highly scalable sketching techniques such as locality-sensitive hashing (LSH) and max-inner product search (MIPS) are available for efficiently finding similar vectors in the embedding space. Last but not least, the semantic embedding approach allows us to straightforwardly generalize the same search algorithm beyond text, such as multi-modality search. For example, can we use a text query to search for images without textual annotations?  Can we search for a website using an image query?  With semantic search, one can simply use the most appropriate vector embedding of these multi-modal objects and jointly train the embeddings using datasets with both text and images.

This tutorial provides you a gentle entry point in deploying AutoMM to semantic search.


```{.python .input}
%%capture
!pip3 install ir_datasets
import ir_datasets
import pandas as pd
pd.set_option('display.max_colwidth', None)
```

### 2. Dataset
In this tutorial, we will use the NF Corpus (Nutrition Facts) dataset from the `ir_datasets` package.
We also convert the query data, document data, and their relevance data into dataframes.

```{.python .input}
%%capture
dataset = ir_datasets.load("beir/nfcorpus/test")

# prepare dataset
doc_data = pd.DataFrame(dataset.docs_iter())
query_data = pd.DataFrame(dataset.queries_iter())
labeled_data = pd.DataFrame(dataset.qrels_iter())
label_col = "relevance"
query_id_col = "query_id"
doc_id_col = "doc_id"
text_col = "text"
id_mappings={query_id_col: query_data.set_index(query_id_col)[text_col], doc_id_col: doc_data.set_index(doc_id_col)[text_col]}
```

The labeled data contain the query ids, document ids, and their relevance scores.
```{.python .input}
labeled_data.head()
```

The query data store the query ids and their corresponding query contents.
```{.python .input}
query_data.head()
```

We need to remove the urls that are not used in search.
```{.python .input}
query_data = query_data.drop("url", axis=1)
query_data.head()
```

The doc data have the document ids as well as the corresponding contents.
```{.python .input}
doc_data.head(1)
```

Similar to the query data, we remove the url column. We also need to concatenate all the valid texts into one column.
```{.python .input}
doc_data[text_col] = doc_data[[text_col, "title"]].apply(" ".join, axis=1)
doc_data = doc_data.drop(["title", "url"], axis=1)
doc_data.head(1)
```

There are 323 queries, 3633 documents, and 12334 relevance scores in the dataset.


### 3. `NDCG@10` Evaluation

Users pay the most attention to the first result, then the second, and etc. 
As a result, precision matters the most for the top-ranked results. 
In this tutorial, we use **Normalized Discounted Cumulative Gain (NDCG)** for measuring the ranking performance.

#### 3.1 Defining CG, DCG, IDCG and NDCG fomulas

In order to understand the NDCG metric, we must first understand CG (Cumulative Gain) and DCG (Discounted Cumulative Gain), as well as understanding the two assumptions that we make when we use DCG and its related measures:

1. Highly relevant documents are more useful when appearing earlier in the search engine results list.
2. Highly relevant documents are more useful than marginally relevant documents, which are more useful than non-relevant documents

First, the primitive **Cumulative Gain (CG)**, which adds the relevance score ($rel$) up to a specified rank position $p$:
$$ \mathrm{CG}_p = \sum_{i=1}^p \mathrm{rel}_i. $$

Then, the **Discounted Cumulative Gain (DCG)**, which penalizes each relevance score logarithmically based on its position in the results:
$$ \mathrm{DCG}_p = \sum_{i=1}^p \frac{\mathrm{rel}_i}{\log_2(i + 1)}. $$

Next, the **Ideal DCG (IDCG)**, which is the DCG of the best possible results based on the given ratings:
$$ \mathrm{IDCG}_p = \sum_{i=1}^{|\mathrm{REL}_p|} \frac{\mathrm{rel}_i}{\log_2(i + 1)}. $$

where $|mathrm{REL}_p|$ is the list of relevant documents (ordered by their relevance) in the corpus up to the position $p$.

And finally, the **NDCG**:
$$ \mathrm{NDCG}_p = \frac{\mathrm{DCG}_p}{\mathrm{IDCG}_p}. $$

We provide an util function to compute the ranking scores. Moreover, we also support measuring NDCG under different cutoffs values.

```{.python .input}
from autogluon.multimodal.utils import compute_ranking_score
cutoffs = [5, 10, 20]
```

### 4. Use BM25

BM25 (or Okapi BM25) is a popular ranking algorithm currently used by OpenSearch for scoring document relevancy to a query. 
We will use BM25's NDCG scores as baselines in this tutorial.

#### 4.1 Defining the formula

$$ score_{BM25} = \sum_i^n \mathrm{IDF}(q_i) \frac{f(q_i, D) \cdot (k1 + 1)}{f(q_i, D) + k1 \cdot (1 - b + b \cdot \frac{fieldLen}{avgFieldLen})}$$

where $\mathrm{IDF}(q_i)$ is the inverse document frequency of the $i^{th}$ query term, and the actual formula used by BM25 for this part is:

$$ \log(1 + \frac{docCount - f(q_i) + 0.5)}{f(q_i) + 0.5}). $$

$k1$ is a tunable hyperparameter that limits how much a single query term can affect the score of a given document. In ElasticSearch, it is [default](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html) to be 1.2.

$b$ is another hyperparameter variable that determines the effect of document length compared to the average document length in the corpus. In ElasticSearch, it is [default](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html) to be 0.75. 

In this tutorial, we will be using the package `rank_bm25` to avoid the complexity of implementing the algorithm from scratch.

#### 4.2 Defining functions


```{.python .input}
%%capture
!pip3 install rank_bm25
from collections import defaultdict
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

nltk.download('stopwords')
nltk.download('punkt')

def tokenize_corpus(corpus):
    stop_words = set(stopwords.words("english") + list(string.punctuation))
    
    tokenized_docs = []
    for doc in corpus:
        tokens = nltk.word_tokenize(doc.lower())
        tokenized_doc = [w for w in tokens if w not in stop_words and len(w) > 2]
        tokenized_docs.append(tokenized_doc)
    return tokenized_docs

def rank_documents_bm25(queries_text, queries_id, docs_id, top_k, bm25):
    tokenized_queries = tokenize_corpus(queries_text)
    
    results = {qid: {} for qid in queries_id}
    for query_idx, query in enumerate(tokenized_queries):
        scores = bm25.get_scores(query)
        scores_top_k_idx = np.argsort(scores)[::-1][:top_k]
        for doc_idx in scores_top_k_idx:
            results[queries_id[query_idx]][docs_id[doc_idx]] = float(scores[doc_idx])
    return results

def get_qrels(dataset):
    """
    Get the ground truth of relevance score for all queries
    """
    qrel_dict = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrel_dict[qrel.query_id][qrel.doc_id] = qrel.relevance
    return qrel_dict

def evaluate_bm25(doc_data, query_data, qrel_dict, cutoffs):
    
    tokenized_corpus = tokenize_corpus(doc_data[text_col].tolist())
    bm25_model = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
    
    results = rank_documents_bm25(query_data[text_col].tolist(), query_data[query_id_col].tolist(), doc_data[doc_id_col].tolist(), max(cutoffs), bm25_model)
    ndcg = compute_ranking_score(results=results, qrel_dict=qrel_dict, metrics=["ndcg"], cutoffs=cutoffs)
    
    return ndcg
```


```{.python .input}
qrel_dict = get_qrels(dataset)
evaluate_bm25(doc_data, query_data, qrel_dict, cutoffs)
```

### 5. Use AutoMM
AutoMM provides easy-to-use APIs to evaluate the ranking performance, extract embeddings, and conduct semantic search.

#### 5.1 Initialize Predictor

For text data, we can initialize `MultiModalPredictor` with problem type `text_similarity`. 
We need to specify `query`, `response`, and `label` with the corresponding column names in the `labeled_data` dataframe.

```{.python .input}
%%capture
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
        query=query_id_col,
        response=doc_id_col,
        label=label_col,
        problem_type="text_similarity",
        hyperparameters={"model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2"}
    )
```

#### 5.2 Evaluate Ranking
Evaluating the ranking performance is easy with the `evaluate` API. 
During evaluation, the predictor automatically extracts embeddings, computes cosine similarities, ranks the results, and computes the scores.
```{.python .input}
predictor.evaluate(
        labeled_data,
        query_data=query_data[[query_id_col]],
        response_data=doc_data[[doc_id_col]],
        id_mappings=id_mappings,
        cutoffs=cutoffs,
        metrics=["ndcg"],
    )
```

We can find significant improvements over the BM25's performances.

#### 5.3 Semantic Search
We also provide an util function for semantic search.

```{.python .input}
from autogluon.multimodal.utils import semantic_search
hits = semantic_search(
        matcher=predictor,
        query_data=query_data[text_col].tolist(),
        response_data=doc_data[text_col].tolist(),
        query_chunk_size=len(query_data),
        top_k=max(cutoffs),
    )
```

We rank the docs according to cosine similarities between the query and document embeddings.
For simplicity, we use `torch.topk` with [linear complexity](https://github.com/pytorch/pytorch/blob/4262c8913c2bddb8d91565888b4871790301faba/aten/src/ATen/native/cuda/TensorTopK.cu#L92-L121) (O(n+k)) to get the k most similar vector embeddings to the query embedding. However, in practice, more efficient methods for similarity search are often used, e.g. [Faiss](https://github.com/facebookresearch/faiss).

#### 5.4 Extract Embeddings
Extracting embeddings is important to deploy models to industry search engines. In general, a system extracts the embeddings for database items offline. During the online search, it only needs to encode query data and then efficiently matches the query embeddings with the saved database embeddings.

```{.python .input}
query_embeds = predictor.extract_embedding(query_data[[query_id_col]], id_mappings=id_mappings, as_tensor=True)
doc_embeds = predictor.extract_embedding(doc_data[[doc_id_col]], id_mappings=id_mappings, as_tensor=True)
```

### 6. Hybrid BM25

We are proposing a new method of search ranking called *Hybrid BM25*, which combines BM25 and semantic embedding for scoring. The key idea is to use BM25 as the first-stage retrieval method (say it recalls 1000 documents for each query), then use a pretrained language model (PLM) to score all the recalled documents (1000 documents). 

We then rerank the retrieved documents with the score calculated as:
$$ score = \beta * normalized\_BM25 + ( 1 - \beta) * score\_of\_plm $$
where 

$$ normalized\_BM25(q_i, D_j) = \frac{\textsf{BM25}(q_i,D_j) - \min_{a\in \mathcal{Q},b\in\mathcal{D}}(\textsf{BM25}(a,b))}{\max_{a\in \mathcal{Q},b\in\mathcal{D}}(\textsf{BM25}(a,b)) - \min_{a\in \mathcal{Q},b\in\mathcal{D}}(\textsf{BM25}(a,b))},$$

and $\beta$ is a tunable parameter, which we will default to $0.3$ in our tutorial.

#### 6.1 Defining functions


```{.python .input}
import torch
from autogluon.multimodal.utils import compute_semantic_similarity

def hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, top_k, beta):
    # Recall documents with BM25 scores
    tokenized_corpus = tokenize_corpus(doc_data[text_col].tolist())
    bm25_model = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
    bm25_scores = rank_documents_bm25(query_data[text_col].tolist(), query_data[query_id_col].tolist(), doc_data[doc_id_col].tolist(), recall_num, bm25_model)
    
    all_bm25_scores = [score for scores in bm25_scores.values() for score in scores.values()]
    max_bm25_score = max(all_bm25_scores)
    min_bm25_score = min(all_bm25_scores)

    q_embeddings = {qid: embed for qid, embed in zip(query_data[query_id_col].tolist(), query_embeds)}
    d_embeddings = {did: embed for did, embed in zip(doc_data[doc_id_col].tolist(), doc_embeds)}
    
    query_ids = query_data[query_id_col].tolist()
    results = {qid: {} for qid in query_ids}
    for idx, qid in enumerate(query_ids):
        rec_docs = bm25_scores[qid]
        rec_doc_emb = [d_embeddings[doc_id] for doc_id in rec_docs.keys()]
        rec_doc_id = [doc_id for doc_id in rec_docs.keys()]
        rec_doc_emb = torch.stack(rec_doc_emb)
        scores = compute_semantic_similarity(q_embeddings[qid], rec_doc_emb)
        scores[torch.isnan(scores)] = -1
        top_k_values, top_k_idxs = torch.topk(
            scores,
            min(top_k + 1, len(scores[0])),
            dim=1,
            largest=True,
            sorted=False,
        )

        for doc_idx, score in zip(top_k_idxs[0], top_k_values[0]):
            doc_id = rec_doc_id[int(doc_idx)]
            # Hybrid scores from BM25 and cosine similarity of embeddings
            results[qid][doc_id] = \
                (1 - beta) * float(score.numpy()) \
                + beta * (bm25_scores[qid][doc_id] - min_bm25_score) / (max_bm25_score - min_bm25_score)
    
    return results


def evaluate_hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, beta, cutoffs):
    results = hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, max(cutoffs), beta)
    ndcg = compute_ranking_score(results=results, qrel_dict=qrel_dict, metrics=["ndcg"], cutoffs=cutoffs)
    return ndcg
```


```{.python .input}
recall_num = 1000
beta = 0.3
query_embeds = predictor.extract_embedding(query_data[[query_id_col]], id_mappings=id_mappings, as_tensor=True)
doc_embeds = predictor.extract_embedding(doc_data[[doc_id_col]], id_mappings=id_mappings, as_tensor=True)
evaluate_hybridBM25(query_data, query_embeds, doc_data, doc_embeds, recall_num, beta, cutoffs)
```

We were able to improve the ranking scores over the naive BM25.

#### 7. Summary

In this tutorial, we have demonstrated how to use AutoMM for semantic search, and showcased the obvious improvements over the classical BM25. We further improved the ranking scores by combining BM25 and AutoMM (Hybrid BM25).
