import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/wordnet')
except LookupError:
    nltk.download('wordnet')

import pandas as pd

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

trec_microblog_src = 'data/Trec_microblog11.txt'
trec_mb_df = pd.read_csv(trec_microblog_src, delimiter='\t', names=["id", "raw_message"])

# Convert all messages to lowercase
trec_mb_df['raw_message'] = trec_mb_df['raw_message'].str.lower()

"""
Produce List of Alphabetic Tokens From Raw Text
"""
def identify_tokens(row):
    message = row['raw_message']
    tokens = nltk.word_tokenize(message)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

"""
Produce Lemmatized List from List of Tokens
"""
def lemmatize_list(row):
    my_list = row['alnum_items']
    lemmatized_list = [lemmatizer.lemmatize(word) for word in my_list] # TODO: use lemmatizer or stemmer or both(maybe only if doesnt end with "e" or "s" then lemma otherwise stem idk lol)
    return (lemmatized_list)

"""
Removes stop words from list of Alphabetic Tokens
"""
def remove_stops(row):
    my_list = row['lemmatized_items']
    meaningful_items = [w for w in my_list if w not in stop_words]
    return (meaningful_items)

"""
Rejoin list back to Space Seperated Strings
"""
def rejoin_words(row):
    my_list = row['message_content']
    joined_words = ( " ".join(my_list))
    return joined_words

# Create column with only alpha numeric tokens
trec_mb_df['alnum_items'] = trec_mb_df.apply(identify_tokens, axis=1)

# Lemmatize words column
trec_mb_df['lemmatized_items'] = trec_mb_df.apply(lemmatize_list, axis=1)

# Removed Stopwords
trec_mb_df['message_content'] = trec_mb_df.apply(remove_stops, axis=1)

# Final Processed Messages
trec_mb_df['processed'] = trec_mb_df.apply(rejoin_words, axis=1)

# Drop Processing Related Columns
cols_to_drop = ['alnum_items', 'lemmatized_items', 'message_content']
trec_mb_df.drop(cols_to_drop, axis=1, inplace=True)

# Save Data to CSV
trec_mb_df.to_csv('data/trec_mb_processed.csv', index=False)

# SOURCE: https://pythonhealthcare.org/2018/12/14/101-pre-processing-data-tokenization-stemming-and-removal-of-stop-words/

