from nltk.corpus import stopwords
import re
import os
import csv
import sys
import nltk
import math
import operator
import datetime
import pandas as pd
from invertedindex import InvertedIndex

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

QUERY_FILE = os.path.join(os.path.dirname(
    __file__), "../data/topics_MB1-49.txt")


class Query:

    @staticmethod
    def get_doc_index():
        inverted_index = InvertedIndex()
        inverted_index.load_serialized()
        return inverted_index

    """
    Query Class
    """

    def __init__(self, raw_query):
        self.query = raw_query
        self.weights = dict()
        self.doc_index = InvertedIndex()
        self.doc_index.load_serialized()

    """
    Process raw query(plain text) and scores it
    """

    def process_raw_query(self):
        self.identify_tokens()
        self.lemmatize_list()
        self.remove_stops()
        self.score()

    """
    Cleans string: lowercase and only alphabetic characters
    """

    def identify_tokens(self):
        self.query = self.query.lower()
        tokens = nltk.word_tokenize(self.query)
        # taken only words (not punctuation)
        self.query = [w for w in tokens if w.isalpha()]

    """
    Applies Lematizer to query
    """

    def lemmatize_list(self):
        self.query = [lemmatizer.lemmatize(word) for word in self.query]

    """
    Removes stopwords from query
    """

    def remove_stops(self):
        self.query = [w for w in self.query if w not in stop_words]

    """
    Scores query: Apply idf-tf to query to produce query vector
    """

    def score(self):
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

    """
    Computes Cosine Similarity between document vector and query vector(self)
    """

    def get_similarity(self, document_weights):
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

    """
    Perfoms query to fetch most similar results
    """

    def perform_query(self, top=1000):
        results = dict()
        for id, vect in self.doc_index.index.items():
            results[id] = self.get_similarity(vect)
        sorted_tuples = sorted(
            results.items(), key=operator.itemgetter(1), reverse=True)
        sorted_results = {k: v for k, v in sorted_tuples}
        top_results = {k: sorted_results[k]
                       for k in list(sorted_results)[:top]}
        return top_results
