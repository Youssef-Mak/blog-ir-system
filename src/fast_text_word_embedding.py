from os import path
import gensim
from nltk.corpus import wordnet


class FastText:

    def __init__(self):
        saved_file_path = path.join(
            path.dirname(__file__), '../data/vectors.vc')
        self.model = None
        if path.exists(saved_file_path):
            print("Detected saved model. Retrieving saved model...")
            self.model = self.load_model(saved_file_path)
        else:
            model_file = path.join(path.dirname(
                __file__), "../data/wiki-news-300d-1M.vec")
            if not path.exists(model_file):
                print(
                    'Looks like wiki-news-300d-1M.vec was not found in the data folder.')
                print(
                    'Please download the wiki-news-300d-1M.vec file from the following link...')
                print(
                    '\n\t\nhttps://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip ')
                raise Exception('Missing pre-trained file')
            print(
                "Since this is your first time initializing the model this might take some time...")
            self.model = gensim.models.KeyedVectors.load_word2vec_format(
                model_file, binary=False, unicode_errors='ignore', limit=500000)
            self.save_model(saved_file_path)
            print('Saved model in ../data/ as vectors.vc for faster loads')

    def save_model(self, file_path):
        print('SAVING...')
        self.model.save(file_path)

    def is_a_synonym_of(self, word_a, word_b):
        synonyms = []
        for syn in wordnet.synsets(word_b):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())

        return word_b in set(synonyms)

    def load_model(self, file_path):
        """
        Returns word vector model that has been saved in the provided file_path
        """
        print('LOADING...')
        return gensim.models.KeyedVectors.load(file_path)

    def similarity_between(self, word1, word2):
        """
        Returns similarity result of provided words based on fast-text pre-trained model on 1 Million wiki news page  
        """
        return self.model.similarity(word1, word2)

    def topNSimilarities(self, word, topN):
        """
        Returns top-n similarity scores for provided word
        """
        if (topN < 1):
            raise Exception('Invalid Argument:\ntopN must be greater then 1')
        return self.model.most_similar(positive=[word], topn=topN)
