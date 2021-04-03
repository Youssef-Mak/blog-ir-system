# Fully Utilizing BERT as Retrieval Model

As an experiment, we chose to use the BERT model without the use of our classical retrieval model.
We mainly used two Pre-Trained BERT Models during this experiment: the stsb-roberta-base and the stsb-roberta-large.

These two models were chosen as candidates as they were especially optimized for semantic text similarity. Both models
were trained in the English language which encompasses the majority of our test query set. They are both pretrained on a large corpus of data
without supervision. 

## Model Details 

Here are the details of the two models:

roberta-base

12-layer, 768-hidden, 12-heads, 125M parameters
RoBERTa using the BERT-base architecture
(see details)

roberta-large

24-layer, 1024-hidden, 16-heads, 355M parameters
RoBERTa using the BERT-large architecture

## Methodology

The approach was quite simple. Firstly, the entire collection of documents is first encoded by the model. This means that the inner document meta-embeddings are
computed. This is by par the most time consuming step as the sheer number of documents is quite large and the models are especially sophisticated with a large number of layers and parameters. This is however circumvented by serializing the model after its initialization, effectively reducing it to a single initialization. The embeddings of the query are then calculated after which the cosine pairwise similarities are computed. Finally, the pairwise similarities are sorted to yield the top results.

## TREC Results

### Using TF-IDF with Cosine Similarity
```
map                     all     0.2075
P_10                    all     0.2408
```

### Using stsb-roberta-base BERT model with Cosine Similarity
```
map                     all     0.0274
P_10                    all     0.0327
```

### Using stsb-roberta-large BERT model with Cosine Similarity
```
map                     all     0.0356
P_10                    all     0.0388
```

## Analysis

Surprisingly, both BERT Models achieved significantly poorer results than our traditional retrieval model. The main potential culprit for this is the selection of the pretrained models. Particularly the chosen models are case-sensitive, making it especially strict relative to our casual microblog collection. We are quite confident that we would have achieved dramatically better results if we'd have chosen a more specific model that is trained more specifically to our collection such as a tweet based model rather than a raw-English Text model such as the models chosen.
