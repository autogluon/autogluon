# Search with Semantic Embedding

### 1. Introduction to semantic embedding

Semantic embedding (e.g., [CLIP](https://openai.com/blog/clip/) or [SentenceTransformer](https://www.sbert.net/)) is one of the main workhorses behind the modern search technology. Instead of directly matching the query to candidates by term frequency (e.g., BM25), a semantic search algorithm matches them by first converting the text $x$ into a feature vector $\phi(x)$ then comparing the similarities using a distance metric defined in that vector space. These feature vectors, known as a "vector embedding", are often trained end-to-end on large text corpus, so that they encode the *semantic* meaning of the text. For example, synonyms are embedded to a similar region of the vector space and relationships between words are often revealed by algebraic operations (see Figure 1 for an example). For these reasons, a vector embedding of text are also known as a **semantic embedding**. With a semantic embedding of the query and the search candidate documents, a search algorithm can often be reduced to finding most similar vectors. This new approach to search is known as **semantic search**.

![Similar sentences have similar embeddings. Image from [Medium](https://medium.com/towards-data-science/fine-grained-analysis-of-sentence-embeddings-a3ff0a42cce5)](https://miro.medium.com/max/1400/0*esMqhzu9WhLiP3bD.jpg)
:width:`500px`

There are three main advantages of using semantic embeddings for a search problem over classical information-retrieval methods (e.g., bag-of-words or TF/IDF).  First, it returns candidates that are related according to the meaning of the text, rather than similar word usage.  This helps to discover paraphrased text and similar concepts described in very different ways. Secondly, semantic search is often more computationally efficient. Vector embeddings of the candidates can be pre-computed and stored in data structures. Highly scalable sketching techniques such as locality-sensitive hashing (LSH) and max-inner product search (MIPS) are available for efficiently finding similar vectors in the embedding space. Last but not least, the semantic embedding approach allows us to straightforwardly generalize the same search algorithm beyond text, such as multi-modality search. For example, can we use a text query to search for images without textual annotations?  Can we search for a website using an image query?  With semantic search, one can simply use the most appropriate vector embedding of these multi-modal objects and jointly train the embeddings using datasets with both text and images.

This tutorial provides you a gentle entry point in learning and deploying state-of-the-art semantic embedding and semantic search.


### 2. Dataset

We will be using the **Financial Question Qnswering (FiQA)** dataset throughout this tutorial for demonstration purpose. Here is an example from the dataset:

<ul>
Q: Is it smarter to buy a small amount of an ETF every 2 or 3 months, instead of monthly?

A: By not timing the market and being a passive investor, the best time to invest is the moment you have extra money (usually when wages are received). The market trends up.  $10 fee on $2000 represents 0.5% transaction cost, which is borderline prohibitive.  I would suggest running simulations, but I suspect that 1 month is the best because average historical monthly total return is more than 0.5%.
</ul>

The dataset is available to load from the `ir_datasets` package.


```{.python .input}
%%capture
!pip3 install ir_datasets
import ir_datasets
import pandas as pd
```


```{.python .input}
%%capture
def map_id_content(ids, contents):
    res = {}
    for id, content in zip(ids, contents):
        res[id] = content
    return res

dataset = ir_datasets.load("beir/fiqa/dev")

# Prepare dataset
docs_df = pd.DataFrame(dataset.docs_iter()).sample(frac=0.2, random_state=42)
queries_df = pd.DataFrame(dataset.queries_iter()).sample(frac=0.2, random_state=42)
docs_text = docs_df["text"].tolist()
docs_id = docs_df["doc_id"].tolist()
queries_text = queries_df["text"].tolist()
queries_id = queries_df["query_id"].tolist()
queries = map_id_content(queries_id, queries_text)
docs = map_id_content(docs_id, docs_text)
```

There are about **57600 documents** and **500 queries** in the dataset.

Here are some more sample questions from the dataset:


```{.python .input}
pd.set_option('display.max_colwidth', None)

queries_df.sample(n=5, random_state=23)
```

As you can tell, financial QA can be sometimes hard to interpret by machine because of many domain specific vocabularies, such as CD (Certificate of Deposit) and ETF (Exchange Traded Fund). Searching for relevant documents for open questions like the above can be challenging with TF/IDF based algorithms (e.g., BM25). In the following sections, we will demonstrate how to perform search rankings based on the BM25 scores and the cosine similarity of semantic embeddings. We will also compare the search quality using different methods with the FiQA dataset.

### 3. `NDCG@10` Evaluation

Users pay the most attention to the first result, then the second, and etc. As a result, precision matters the most for the top-ranked results. In this tutorial, we use **Normalized Discounted Cumulative Gain (NDCG)** for measuring the quality of the top 10 ranked results (a.k.a., **NDCG@10**).

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

In this tutorial, we will be using the python package `pytrec_eval` for evaluating `NDCG@10`.

#### 3.2 Defining functions


```{.python .input}
%%capture
from collections import defaultdict

!pip3 install pytrec_eval
import pytrec_eval

def ir_metrics(qrel_dict, results, cutoff=[10]):
    """Get the NDCG

    Parameters
    ----------
    qrel_dict:
        the groundtruth query and document relavance
    results:
        the query/document ranking list by the model
    cutoff:
        the position cutoff for NDCG evaluation
    """
    ndcg = {}
    for k in cutoff:
        ndcg[f"NDCG@{k}"] = 0.0

    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in cutoff])

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel_dict, {ndcg_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in cutoff:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]

    for k in cutoff:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)

    return ndcg

def get_qrels(dataset):
    """
    Get the ground truth of relevance score for all queries
    """
    qrel_dict = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrel_dict[qrel.query_id][qrel.doc_id] = qrel.relevance
    return qrel_dict
```

### 4. Use BM25

BM25 (or Okapi BM25) is a popular ranking algorithm currently used by OpenSearch for scoring document relevancy to a query. We will use the `NDCG@10` score for BM25 as a baseline in this tutorial.

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

def evaluate_bm25(docs_text, queries_text, docs_id, queries_id, qrel_dict, top_k):
    tokenized_corpus = tokenize_corpus(docs_text)
    bm25_model = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
    
    results = rank_documents_bm25(queries_text, queries_id, docs_id, top_k, bm25_model)
    ndcg = ir_metrics(qrel_dict, results)
    
    print("NDCG@10 for BM25: ", ndcg["NDCG@10"])
    return ndcg
```


```{.python .input}
qrel_dict = get_qrels(dataset)
evaluate_bm25(docs_text, queries_text, docs_id, queries_id, qrel_dict, 10)
```

The `NDCG@10` when using BM25 is 0.2231.

### 5. Use Sementic Embedding

We rank the documents according to cosine similarities between the query embedding and the document embeddings. `NDCG@10` is evaluated based on all the queries in the dataset.

#### 5.1 Extracting Embedding

AutoGluon's `MultiModalPredictor` provides a nice interface for extracting embeddings with a pretrained model by using the `feature_extraction` pipeline. It extracts the hidden states from the base model, which then can be used to extract embeddings for queries and documents.


```{.python .input}
%%capture
from autogluon.multimodal import MultiModalPredictor

model_name = "sentence-transformers/all-MiniLM-L6-v2"

predictor = MultiModalPredictor(
    pipeline="feature_extraction",
    hyperparameters={
        "model.hf_text.checkpoint_name": model_name
    }
)
```

Then we use `predictor.extract_embedding(docs_df)` to get the embeddings.

#### 5.2 Defining functions

For illustration purpose, we use `torch.topk` with [linear complexity](https://github.com/pytorch/pytorch/blob/4262c8913c2bddb8d91565888b4871790301faba/aten/src/ATen/native/cuda/TensorTopK.cu#L92-L121) (O(n+k)) to get the k most similar vector embeddings to the query embedding. However, in practice, more efficient methods for similarity search are often used, e.g. [Faiss](https://github.com/facebookresearch/faiss).

```{.python .input}
from collections import defaultdict

import torch

def cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all a[i] and b[j].
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def extract_embedding(texts, predictor):
    embeddings = predictor.extract_embedding(texts)["0"]
    return embeddings

def get_top_k(scores, top_k):
    scores[torch.isnan(scores)] = -1
    top_k_values, top_k_idx = torch.topk(
        scores,
        min(top_k + 1, len(scores[0])),
        dim=1,
        largest=True,
        sorted=False,
    )
    return top_k_values, top_k_idx

def evaluate_semantic_embedding(docs_embeddings, queries_embeddings, docs_id, queries_id, qrel_dict, top_k):
    scores = cos_sim(queries_embeddings, docs_embeddings)
    scores_top_k_values, scores_top_k_idx = get_top_k(scores, top_k)
    
    results = {qid: {} for qid in queries_id}
    for query_idx in range(len(queries_id)):
        for doc_idx, score in zip(scores_top_k_idx[query_idx], scores_top_k_values[query_idx]):
            results[queries_id[query_idx]][docs_id[int(doc_idx)]] = float(score.numpy())
    
    ndcg = ir_metrics(qrel_dict, results)
    print("NDCG@10 for semantic embedding: ", ndcg["NDCG@10"])
    return ndcg
```


```{.python .input}
%%capture
# extract embeddings for corpus and queries
queries_embeddings = extract_embedding(queries_text, predictor)
docs_embeddings = extract_embedding(docs_text, predictor)
queries_embeddings = queries_embeddings if isinstance(queries_embeddings, torch.Tensor) else torch.from_numpy(queries_embeddings)
docs_embeddings = docs_embeddings if isinstance(docs_embeddings, torch.Tensor) else torch.from_numpy(docs_embeddings)
```

```{.python .input}
evaluate_semantic_embedding(docs_embeddings, queries_embeddings, docs_id, queries_id, qrel_dict, 10)
```

The `NDCG@10` when using semantic embedding is 0.38061. There is already a significant improvement over the `NDCG@10` when we used BM25 as the scoring function.

### 6. Hybrid BM25

We are proposing a new method of search ranking called *Hybrid BM25*, which combines BM25 and semantic embedding for scoring. The key idea is to use BM25 as the first-stage retrieval method (say it recalls 1000 documents for each query), then use a pretrained language model (PLM) to score all the recalled documents (1000 documents). 

We then rerank the retrieved documents with the score calculated as:
$$ score = \beta * normalized\_BM25 + ( 1 - \beta) * score\_of\_plm $$
where 

$$ normalized\_BM25(q_i, D_j) = \frac{\textsf{BM25}(q_i,D_j) - \min_{a\in \mathcal{Q},b\in\mathcal{D}}(\textsf{BM25}(a,b))}{\max_{a\in \mathcal{Q},b\in\mathcal{D}}(\textsf{BM25}(a,b)) - \min_{a\in \mathcal{Q},b\in\mathcal{D}}(\textsf{BM25}(a,b))},$$

and $\beta$ is a tunable parameter, which we will default to $0.3$ in our tutorial.

#### 6.1 Defining functions


```{.python .input}
def hybridBM25(docs_text, queries_text, docs_id, queries_id, recall_num, top_k, beta):
    # Recall documents with BM25 scores
    tokenized_corpus = tokenize_corpus(docs_text)
    bm25_model = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)
    bm25_scores = rank_documents_bm25(queries_text, queries_id, docs_id, recall_num, bm25_model)
    
    all_bm25_scores = [score for scores in bm25_scores.values() for score in scores.values()]
    max_bm25_score = max(all_bm25_scores)
    min_bm25_score = min(all_bm25_scores)

    q_embeddings = map_id_content(queries_id, queries_embeddings)
    d_embeddings = map_id_content(docs_id, docs_embeddings)
    
    results = {qid: {} for qid in queries_id}
    for idx, qid in enumerate(queries_id):
        rec_docs = bm25_scores[qid]
        rec_doc_emb = [d_embeddings[doc_id] for doc_id in rec_docs.keys()]
        rec_doc_id = [doc_id for doc_id in rec_docs.keys()]
        rec_doc_emb = torch.stack(rec_doc_emb)
        scores = cos_sim(q_embeddings[qid], rec_doc_emb)
        scores_top_k_values, scores_top_k_idx = get_top_k(scores, top_k)

        for doc_idx, score in zip(scores_top_k_idx[0], scores_top_k_values[0]):
            doc_id = rec_doc_id[int(doc_idx)]
            # Hybrid scores from BM25 and cosine similarity of embeddings
            results[qid][doc_id] = \
                (1 - beta) * float(score.numpy()) \
                + beta * (bm25_scores[qid][doc_id] - min_bm25_score) / (max_bm25_score - min_bm25_score)
    
    return results


def evaluate_hybridBM25(recall_num, beta, top_k):
    results = hybridBM25(docs_text, queries_text, docs_id, queries_id, recall_num, top_k, beta)
    ndcg = ir_metrics(qrel_dict, results)
    
    print("NDCG@10 for Hybrid BM25: ", ndcg["NDCG@10"])
    return ndcg
```


```{.python .input}
recall_num = 1000
beta = 0.3
top_k = 10
evaluate_hybridBM25(recall_num, beta, top_k)
```

We were able to improve the `NDCG@10` score from 0.38061 to 0.38439 by tuning `beta` and `recall_num`.

#### 7. Summary

In this tutorial, we have demonstrated in details how different search ranking methods are implemented, and showcased the drastic improvement of semantic embedding over the classical BM25 algorithm. We further improved the `NDCG@10` score by combining BM25 and semantic embedding (Hybrid BM25) and tuning the hyperparameters.
