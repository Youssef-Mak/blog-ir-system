[![Run on Repl.it](https://repl.it/badge/github/Youssef-Mak/blog-ir-system)](https://repl.it/@YoussefMak1/blog-ir-system)
# blog-ir-system

Information Retrieval (IR) system based for a collection of documents (Twitter messages)

## Setup 
Install the requirements 
```
pip install -r requirements.txt
```

## Preprocessing
```
python src/preprocessing.py
```

## Building an Inverted Index 
```
python src/invertedindex.py
```

## Running a Query 

### Individual Query
```
python src/query.py "<query>"
```

### Batch Query
Default batch defined in `data/topics_MB1-49.txt`
```
python src/query.py 
```
