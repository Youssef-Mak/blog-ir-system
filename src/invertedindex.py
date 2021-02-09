import csv
import math


class InvertedIndex:
    """
    InvertedIndex Class 
    """
    def __init__(self):
        self.index = dict()
        self.db = dict()

    """
    Parse CSV file with a "id" and "processed" columns to generate a dict
    """
    def parse_csv(self, input_csv='data/trec_mb_processed.csv'):
        with open(input_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.db[row['id']] = row['processed']

    """
    Construct Inverted Index
    """
    def construct_index(self):
        total_freq = dict()
        doc_size = len(self.db.keys())
        for id in self.db:
            term_list = self.db[id].split(' ')
            weights = {}
            for term in term_list:
                if term not in weights:
                    document_freq = term_list.count(term)
                    weights[term] = document_freq
                    if term in total_freq:
                        total_freq[term] = total_freq[term] + document_freq
                    else:
                        total_freq[term] = document_freq
            self.index[id] = weights
        for id, document_freq in self.index.items():
            max_freq = max(document_freq.values())
            for term in document_freq:
                # tfi * idfi
                tf = document_freq[term] / max_freq
                idf = math.log(doc_size / total_freq[term])
                document_freq[term] = tf * idf



