[![Run on Repl.it](https://repl.it/badge/github/Youssef-Mak/blog-ir-system)](https://repl.it/@YoussefMak1/blog-ir-system)
# Tweet Search Engine

Information Retrieval (IR) system based for a collection of documents (Twitter messages)

## Setup 
Install [anaconda](https://docs.anaconda.com/anaconda/install/) and activate with `conda activate`.

Install the requirements 
```
pip install -r requirements.txt 
```

Fetch the Pre-trained Vector Model

```
sh init.sh
```

## Preprocessing
```
python src/preprocessing.py
```

## Building an Inverted Index 
```
python src/inverted_index.py
```

## Usage

### General Usage

```
usage: main.py [-h] [-Q QUERY] [-F] [-M {default,BERT}] [-QE {level-1,level-2,none}]

optional arguments:
  -h, --help            show this help message and exit
  -Q QUERY, --query QUERY
                        string to query
  -F, --file            Run queries defined in ../data/topics_MB1-49.txt
  -M {default,BERT}, --method {default,BERT}
                        Neural Method for Retrieval
  -QE {level-1,level-2,none}, --queryexpansion {level-1,level-2,none}
                        Level of query expansion
```

### Running a Query

Note:`default` corresponds to tf-idf weighted cosine similarity retrieval method

#### Individual Query
To run specified query in command and output results in console:
```
python src/main.py -M {BERT|default} -Q "<query>" 
```

#### Batch Query
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

### Using TF-IDF with Cosine Similarity and Query Expansion Level 1
```
map                   	all	0.2070
P_10                  	all	0.2408
```

### Using TF-IDF with Cosine Similarity and Query Expansion Level 2
```
map                   	all	0.2070
P_10                  	all	0.2408
```

