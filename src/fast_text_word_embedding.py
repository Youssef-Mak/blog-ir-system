from os import path
# import fasttext
import gensim
from nltk.corpus import wordnet

# dont forget to also add python-Levenshtein 0.12.2 for to get rid of error message

class FastText ():
    def __init__(self):
        saved_file_path = path.join( path.dirname( __file__),'../data/vectors.vc')
        self.model = None
        if (path.exists ( saved_file_path )):
            print ("Detected saved model. Retrieving saved model...")
            self.model = self.load_model (saved_file_path)
        else:
            fastTextModelPath =  path.join ( path.dirname (__file__),"../data/wiki-news-300d-1M.vec" ) 
            #if pre trained model does not exist then download from link
            if ( not path.exists (fastTextModelPath) ):
                print ('Looks like wiki-news-300d-1M.vec was not found in the data folder.\nPlease download the wiki-news-300d-1M.vec file from the following link...\n\t\nhttps://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip ')
                raise Exception('Missing pre-trained file')
            model_file = fastTextModelPath
            print("remember to remove limit when pushing to repo")
            print ("Since this is your first time initializing the model this might take some time...")
            self.model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False, unicode_errors='ignore', limit=500000)
            #save model for faster loads 
            self.save_model (saved_file_path)
            print ('Saved model in ../data/ as vectors.vc for faster loads')
    
    def save_model (self, file_path):
        print ('SAVING...')
        self.model.save (file_path)
    
    def is_a_synonym_of (self, word1:str, word2:str):
        synonyms = []
        for syn in wordnet.synsets(word2):
            for lm in syn.lemmas():
                synonyms.append(lm.name())#adding into synonyms
        
        return word1 in set(synonyms)

    '''
    Returns word vector model that has been saved in the provided file_path
    '''
    def load_model (self,file_path):
        print ('LOADING...')
        return gensim.models.KeyedVectors.load (file_path)
        
    '''
    Returns similarity result of provided words based on fast-text pre-trained model on 1 Million wiki news page  
    '''
    def similarity_between (self, word1:str, word2:str):
        return self.model.similarity (word1, word2)
    '''
    Returns top-n similarity scores for provided word
    '''
    def topNSimilarities (self, word:str, topN: int):
        if (topN < 1):
            raise Exception ('Invalid Argument:\ntopN must be greater then 1')
        else:
            return self.model.most_similar (positive=[word],topn=topN)
