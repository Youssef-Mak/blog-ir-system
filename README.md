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
To run specified query in command and output results in console:
```
python src/query.py "<query>"
```

### Batch Query
To run default batch defined in `data/topics_MB1-49.txt` and produce `results.txt`:
```
python src/query.py 
```
