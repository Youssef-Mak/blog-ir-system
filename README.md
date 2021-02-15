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

## Trec Results
```
map                     all     0.2075
P_10                    all     0.2408
```

## Functionality
Our Microblog information retrieval system is built using Python. Python is a general-purpose programming language that is interpreted(runs line by line, rather than compilation), dynamically-typed and emphasizes simplicity through object orientation and clean syntax. The language is utilized in various fields such as web applications, data science, artificial intelligence and most importantly natural language processing(NLP). For our project, we used Python's famous NLP library called Natural Language Tool Kit(NLTK), pandas library for reading and writing text-CSV files, and pickles library for serializing and deserialization of our inverted index. The IR system code consists of only 3 files: preprocessing.py, invertedindex.py and query.py. The preprossesing.py file deals with all the preprocessing stages in step 1. Furthermore, the invertedindex.py constructs the inverted index specified in step 2, using the dictionary data structure(Pythons Key-Value pair Data structure). Additionally, the invertedindex.py deals with serializing and deserializing the inverted index to the local machine. Lastly, the main program of the IR system is executed through query.py by passing a query in its argument to generate top 10-results in console or passing no argument to read from query file and generate top 1000-results in a Results.txt file.

The IR system first parses the Trec_microblog11.txt, extracting the tweet-id and raw-data. It then proceeds to process the raw-data by firstly converting the message to lowercase, secondly tokenizing the terms, thirdly applying lemmatization to tokens and finally removing stop words. After processing is finished, it then stores the processed data in a CSV file(trec_mb_processed.csv), alongside its corresponding tweet-id and raw-data.

During the construction of the inverted index, we first parse the trec_mb_processed.csv file to retrieve the id, processed message and raw message. We then create two python dictionaries to store the processed data in one and the raw data in the other with their corresponding id's. To construct the inverted index, we first go through each tweet in the processed data dictionary, extracting every term in the processed message into a list called term_list. In layman terms, the variable term_list contains all the processed terms/tokens per each tweet. Then for each term, we calculate and store its frequency within the tweet, in parallel we also calculate and store the total frequency of the term/token in the corpus. To calculate the weight(t_fi * idf_i) of each term occurring frequently in the tweet we used the following formula...

>$f_ij =$ frequency of term i in document j
>$tf_ij = fij / max_i {f_ij}$ (normalization across corpus)
>$df_i = $ number of documents containing term i
>$idf_i = $ inverse document frequency of term $i,= log2(total_number_of_documents/ dfi)$
>$w_ij = tf_ij ∙ idf_i = tf_ij ∙ log2($total number of documents$/ d_fi)$
>
>Figure 1. Document TF ∙ IDF Equations & Calculations

Additionally, once the inverse index is successfully built it is also serialized locally using python's pickle library for non-repetitve, faster construction and retrieval.

The main component of the IR system is handled in the query stage. In its basic form, the program accepts a string that is provided in the argument before code execution(eg. python query.py "enter any query"). Similarly to the inverted index, the query is then tokenized, lemmatized and stop-words are removed. Furthermore, the query term tf-idf weights are calculated and are stored in a dictionary(key=term, value= $(0.5 + 0.5∙tf_iq) ∙ idf_i)$ however, f_iq is calculated in respect to frequency of term i in query q rather then document. Therefore, to calculate the weifght of each term in respect to the query we utlilize a slightly different formula then the one used in Figure 1...

>$f_iq =$ frequency of term $i$ in query $q$
>$tf_iq = f_iq / max_i {f_iq}$ (normalization across corpus)
>$df_i =$ number of documents containing term $i$
>$idf_i =$ inverse document frequency of term $i$, $= log2($total number of documents$/ df_i)$
>$wiq = (0.5 + 0.5 tf_iq) ∙ idf_i = (0.5 + 0.5 tf_iq) ∙ log2($total number of documents$/ df_i)$ 
>
>Figure 2. Query TF ∙ IDF Equations & Calculation

In Figure 2, the tf_iq is multiplied and added by $1/2$ to provide a slightly better results for the query in comparison to using the formula in Figure 1. For the retrieval process we simply go through the index, calculate and store the cosine similarity between the tweet and query in a results dictionary. For the results process we sort the results dictionary by highest similarity results and return the specified amount(default=1000) results.

## Algorithm & Data Structure
Throughout the course of the project Python dictionaries(key-value pair Data structure) were the most commonly used Data Structure, followed by lists for basic storing needs. The preprocessing component(```preprocessing.py```) utilizes a dictionary-like data structure to parse and manipulate the data in ```Trec_microblog11.txt```. Additionally, the inverted index component(```invertedindex.py```) utilizes dictionaries for the inverted index and other related data such as term weights, total frequency and the raw messages. Lastly, the query component(```query.py```) utilized dictionaries to store the term weights and results from the retrieval process. The main use of dictionaries was its ease of manipulating/storing key-value pairs and rapid retrieval process.

To process our data we used pythons NLTK(Natural Language Tool Kit) library which provides various functions/modules such as WordNetLemmatizer, PorterStemmer and stopwords etc. To improve results, WordNetLemmatizer was utilized for its intelligent operation; which takes a word as its input and outputs the root word(lemma). Utilizing lemmatization offered higher consistency and low ambiguity in the output because the lemma is verified using Princeton University's famous wordnet corpus. For example, using the PorterStemmer stemming algorithm versus WordNetLemmatizer...

>```'studies'```    
>```PorterStemmer $->$ 'studi'```
>```WordNetLemmatizer $->$ 'study'```
>Figure 3. Stemming vs Lemmatization

in Figure 3 showcases an example where the stemming algorithm provides an incorrect word by removing the affixes and replacing with 'i', whereas the root word outputted by lemmatizer is correct. Additionaly, for similarity calculations Cosine Similarity algorithm and equation was used rather then inner product.

## Vocabulary & Example
Total Vocab: 54045

### 100 Vocab token samples

1. 'save'
2. 'bbc'
3. 'world'
4. 'service'
5. 'savage'
6. 'cut'
7. 'http'
8. 'lot'
9. 'people'
10. 'always'
11. 'make'
12. 'fun'
13. 'end'
14. 'question'
15. 'u'
16. 'ready'
17. 'rethink'
18. 'group'
19. 'positive'
20. 'outlook'
21. 'technology'
22. 'staffing'
23. 'specialist'
24. 'expects'
25. 'revenue'
26. 'marg'
27. 'fund'
28. 'manager'
29. 'phoenix'
30. 'appoints'
31. 'new'
32. 'ceo'
33. 'buy'
34. 'closed'
35. 'business'
36. 'latest'
37. 'top'
38. 'release'
39. 'cdt'
40. 'present'
41. 'alice'
42. 'wonderland'
43. 'catonsville'
44. 'dinner'
45. 'ha'
46. 'posted'
47. 'territory'
48. 'location'
49. 'calgary'
50. 'alberta'
51. 'canada'
52. 'job'
53. 'category'
54. 'bu'
55. 'news'
56. 'today'
57. 'free'
58. 'school'
59. 'funding'
60. 'plan'
61. 'transparency'
62. 'manchester'
63. 'city'
64. 'council'
65. 'detail'
66. 'saving'
67. 'depressing'
68. 'apparently'
69. 'deprived'
70. 'hardest'
71. 'hit'
72. 'interested'
73. 'professional'
74. 'global'
75. 'translation'
76. 'fitness'
77. 'first'
78. 'float'
79. 'full'
80. 'model'
81. 'dead'
82. 'david'
83. 'cook'
84. 'mostest'
85. 'beautiful'
86. 'smile'
87. 'piss'
88. 'cnt'
89. 'stand'
90. 'lick'
91. 'ass'
92. 'beware'
93. 'blue'
94. 'meany'
95. 'thebluemeanies'
96. 'como'
97. 'perde'
98. 'dentes'
99. 'warcraft'
100. 'via'

### 10 Answers to Query 3 and Query 20
``` python3 src/query.py 'Haiti Aristide return'```

			TOP TEN RESULTS

			ID: 29296574815272960


			Raw Message: haiti – aristide : his return, an international affair… – haitilibre.com http://bit.ly/gzylxg #haiti


			Similarity: 0.8623073191139999



			ID: 32204788955357184


			Raw Message: haiti opens door for return of ex-president aristide http://tf.to/fjdt


			Similarity: 0.7336347804980093



			ID: 29613127372898304


			Raw Message: if duvalier can return to haiti, why can’t aristide? – new america media http://bit.ly/ecwstk #haiti


			Similarity: 0.7287186959543963



			ID: 32333726654398464


			Raw Message: #aristide!!


			Similarity: 0.7065208548053392



			ID: 29278582916251649


			Raw Message: haiti - aristide : his return, an international affair... - http://haitilibre.com/fben.php?id=2193


			Similarity: 0.6960319269032201



			ID: 33711164877701120


			Raw Message: haiti's former president jean-bertrand aristide vows to return http://gu.com/p/2nvx3/tf


			Similarity: 0.6536217298987377



			ID: 29615296666931200


			Raw Message: if duvalier can return to haiti, why can't aristide? - new america media http://goo.gl/fb/uskrk


			Similarity: 0.6306349217090794



			ID: 32383831071793152


			Raw Message: yah haiti: haiti allows ex-president's return: jean-bertrand aristide, who was haiti's first democratically ele.... http://bit.ly/hlagwo


			Similarity: 0.6030467225450756



			ID: 32273316047757312


			Raw Message: haiti to give aristide passport - http://www.bbc.co.uk/news/world-latin-america-12330414


			Similarity: 0.5974893449963001



			ID: 32211683082502144


			Raw Message: #int'l #news: haiti opens door for return of ex-president aristide: port-au-prince (reuters) - haiti'... http://bit.ly/gsifwd #singapore


			Similarity: 0.5737569283251074 


```		python3 src/query.py 'Taco Bell filling lawsuit' ```

			TOP TEN RESULTS

			ID: 32218912527482880


			Raw Message: lawsuit to taco bell: where?s the beef? http://daily.rssnewest.com/lawsuit-to-taco-bell-wheres-the-beef/


			Similarity: 0.6349187244712484



			ID: 31161931205181440


			Raw Message: eating meat filling all 35% of it (@ taco bell) http://4sq.com/fsf4es


			Similarity: 0.6327628393069771



			ID: 30283063699177472


			Raw Message: oh, my... taco bell's "taco meat filling" is only 36% beef... http://gizmodo.com/5742413/ #fb


			Similarity: 0.61362354820435



			ID: 34082003779330048


			Raw Message: taco bell :)


			Similarity: 0.5834689425972924



			ID: 32899186038935552


			Raw Message: taco bell ^_^


			Similarity: 0.5834689425972924



			ID: 32672996137111552


			Raw Message: @h1gherone this taco bell


			Similarity: 0.5834689425972924



			ID: 32269178773708800


			Raw Message: taco bell :)


			Similarity: 0.5834689425972924



			ID: 31094164959531009


			Raw Message: taco bell.


			Similarity: 0.5834689425972924



			ID: 31038292132634624


			Raw Message: taco bell


			Similarity: 0.5834689425972924



			ID: 31484954504331264


			Raw Message: why am i here? :/ (@ taco bell) http://4sq.com/egft27


			Similarity: 0.5813931115652652 

## Final Results Discussion
Through observation query 3 outputted a high of 86% similarity score whereas query 20 outputted a high of 63%, therefore, the output for query 3 was more precise in terms of similarity results. There could be a number of factors in receiving lower similarity scores for query 20; maybe our corpus lacked data in regards to the query or the retrieval and ranking process needs some improvement. However, even a similarity result of 63% is still very ideal, as the top-ranked result contained the information requested in the query. To slightly further improve results, one may experiment and implement the use of both stemming and lemmatization.