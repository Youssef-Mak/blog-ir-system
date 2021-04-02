import re
import os
import csv
import sys
import nltk
import math
import operator
import datetime
import pickle
import pandas as pd
from nltk.corpus import stopwords
from inverted_index import InvertedIndex
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.model_utils import get_inverted_index, get_bert_embeddings

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))
sbert_model = SentenceTransformer('stsb-roberta-large')


class Query:
    """Class representing Query Object

    Attributes:
        query (str|[]): Query.
        weights (dict): map containing td-idf weights for each term in query
        doc_index (inverted_index): InvertedIndex corresponding to list of indexed documents
    """

    def __init__(self, raw_query):
        """
        Query Class
        """
        self.query = raw_query
        self.weights = dict()
        self.doc_index = get_inverted_index()

    def process_raw_query(self):
        """
        Process raw query(plain text) and scores it
        """
        self.identify_tokens()
        self.lemmatize_list()
        self.remove_stops()
        self.score()

    def identify_tokens(self):
        """
        Cleans string: lowercase and only alphabetic characters
        """
        self.query = self.query.lower()
        tokens = nltk.word_tokenize(self.query)
        # taken only words (not punctuation)
        self.query = [w for w in tokens if w.isalpha()]

    def lemmatize_list(self):
        """
        Applies Lematizer to query
        """
        self.query = [lemmatizer.lemmatize(word) for word in self.query]

    def remove_stops(self):
        """
        Removes stopwords from query
        """
        self.query = [w for w in self.query if w not in stop_words]

    def score(self):
        """
        Scores query: Apply idf-tf to query to produce query vector
        """
        doc_size = len(self.doc_index.db.keys())
        for term in self.query:
            if term not in self.weights:
                self.weights[term] = self.query.count(term)
        max_freq = max(self.weights.values())
        for term in self.weights:
            # tfi * idfi
            tf = self.weights[term] / max_freq
            idf = 0 if term not in self.doc_index.total_freq else math.log(
                doc_size / self.doc_index.total_freq[term])
            self.weights[term] = (0.5 + (0.5 * tf)) * idf

    def get_similarity(self, document_weights):
        """
        Computes Cosine Similarity between query vector(self) and document vector
        """
        numerator = 0
        wiq = 0
        wid = 0
        for term, weight in self.weights.items():
            numerator = numerator if term not in document_weights else numerator + \
                (weight * document_weights[term])
            wiq = wiq + math.pow(weight, 2)
        for term, weight in document_weights.items():
            wid = wid + math.pow(weight, 2)
        return numerator / math.sqrt(wiq*wid)

    def perform_query(self, method='default', top=1000):
        """
        Perfoms query to fetch most similar results
        """
        results = {}
        if method == 'default':
            results = dict()
            for id, vect in self.doc_index.index.items():
                results[id] = self.get_similarity(vect)
        elif method == 'BERT':
            ids = []
            documents = []
            for id, document in self.doc_index.db.items():
                ids.append(id)
                documents.append(document)
            data = {'id': ids,
                    'documents': documents}
            df = pd.DataFrame.from_dict(data)
            query_embeddings = sbert_model.encode(self.query)
            document_embeddings = get_bert_embeddings()
            pairwise_similarities = cosine_similarity(
                query_embeddings, document_embeddings)[0]
            results = dict(zip(df['id'], pairwise_similarities))

        sorted_tuples = sorted(
            results.items(), key=operator.itemgetter(1), reverse=True)
        sorted_results = {k: v for k, v in sorted_tuples}
        top_results = {k: sorted_results[k]
                       for k in list(sorted_results)[:top]}
        return top_results
