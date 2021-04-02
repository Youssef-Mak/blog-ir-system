import os
import pickle
import pandas as pd
from inverted_index import InvertedIndex
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('stsb-roberta-large')

BERT_PATH = os.path.join(os.path.dirname(
    __file__), "../../indexes/sbert_model.pkl")


def get_inverted_index():
    """
    Fetches Serialized Inverted Index
    """
    inverted_index = InvertedIndex()
    inverted_index.load_serialized()
    return inverted_index


def get_bert_embeddings():
    """
    Fetches Serialized Large BERT Embeddings
    """
    with open(BERT_PATH, "rb") as input_file:
        loaded_bert_embeddings = pickle.load(input_file)
        return loaded_bert_embeddings
    return generate_bert_embeddings()


def generate_bert_embeddings():
    """
    Generate BERT Embeddings
    """
    doc_index = get_inverted_index()
    ids = []
    documents = []
    for id, document in doc_index.db.items():
        ids.append(id)
        documents.append(document)
    data = {'id': ids,
            'documents': documents}
    df = pd.DataFrame.from_dict(data)
    documents = df['documents']
    document_embeddings = sbert_model.encode(documents)
    with open(BERT_PATH, "wb") as output_file:
        pickle.dump(document_embeddings, output_file)
    return document_embeddings
