import csv
import math
import pickle
import os

INDEX_PATH = os.path.join(os.path.dirname(__file__), "../indexes/inverted_index.pkl")


class InvertedIndex:
    """
    InvertedIndex Class
    """
    def __init__(self):
        self.index = dict()
        self.db = dict()
        self.raw_db = dict()
        self.total_freq = dict()

    """
    Loads serialized Inverted Index 
    """
    def load_serialized(self):
        with open(INDEX_PATH, "rb") as input_file:
            loaded = pickle.load(input_file)
            self.index = loaded.index
            self.db = loaded.db
            self.total_freq = loaded.total_freq
            self.raw_db = loaded.raw_db

    """
    Parse CSV file with a "id" and "processed" columns to generate a dict
    """
    def parse_csv(self, input_csv='data/trec_mb_processed.csv'):
        with open(input_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.db[row['id']] = row['processed']
                self.raw_db[row['id']] = row['raw_message']

    """
    Construct Inverted Index
    """
    def construct_index(self):
        self.total_freq = dict()
        doc_size = len(self.db.keys())
        for id in self.db:
            term_list = self.db[id].split(' ')
            weights = dict()
            for term in term_list:
                if term not in weights:
                    document_freq = term_list.count(term)
                    weights[term] = document_freq
                    if term in self.total_freq:
                        self.total_freq[term] = self.total_freq[term] + document_freq
                    else:
                        self.total_freq[term] = document_freq
            self.index[id] = weights
        for id, document_freq in self.index.items():
            max_freq = max(document_freq.values())
            for term in document_freq:
                # tfi * idfi
                tf = document_freq[term] / max_freq
                idf = math.log(doc_size / self.total_freq[term])
                document_freq[term] = tf * idf


"""
Produce Inverted Index and Serialize 
"""
if __name__ == "__main__":
    inverted_index = InvertedIndex()
    inverted_index.parse_csv()
    inverted_index.construct_index()
    with open(INDEX_PATH, "wb") as output_file:
        pickle.dump(inverted_index, output_file)


