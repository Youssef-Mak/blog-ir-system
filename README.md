[![Run on Repl.it](https://repl.it/badge/github/Youssef-Mak/blog-ir-system)](https://repl.it/@YoussefMak1/blog-ir-system)
# Tweet Search Engine

Information Retrieval (IR) system based for a collection of documents (Twitter messages)

## Setup 
Install the requirements (NOTE: if this step doesn't work, simply install [anaconda](https://docs.anaconda.com/anaconda/install/))
```
pip install -r requirements.txt
```

## Preprocessing
```
python src/preprocessing.py
```

## Building an Inverted Index 
```
python src/inverted_index.py
```

## Running a Query

Note:`default` corresponds to tf-idf weighted cosine similarity retrieval method

### Individual Query
To run specified query in command and output results in console:
```
python src/main.py -M {BERT|default} -Q "<query>" 
```

### Batch Query
To run default batch defined in `data/topics_MB1-49.txt` and produce `Results.txt`:
```
python src/main.py -M {BERT|default} -F 
```

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

