## QUERY Expansion

### Using TF-IDF with Cosine Similarity and Query Expansion Level 1
```
map                   	all	0.2070
P_10                  	all	0.2408
```

### Using TF-IDF with Cosine Similarity and Query Expansion Level 2
```
map                   	all	0.2070
P_10                  	all	0.2408
```

##Experiment 2
The algorithm for query expansion has 3 options(none, level-1 and level-2). When none is selected no query expansion takes place. 

For option  level-1 the algorithm is as follows…
```
	for each doc:
		for each vector in doc:
			unwanted_terms={set of terms we don’t want in query}
			for each index_term in the vector:
				if index_term does NOT exist in un_wanted_terms:
					similarity_counter=0 //count the number similar index terms matches with query terms
					for each query_iterm in the query:
						similarity score = caculate similarity between index_term and query_term
						if (the similarity score>65% ):
							similarity_counter++//increment similarity counter
					if (imilarity_counter>1):
						add index_item to query
					else:
						add index_item to unwanted_terms //this is important as we don’t want to keep calculating similarity results for words which we have done already
```
For option  level-2 the algorithm is as follows…
```	
    for each doc:
		for each vector in doc:
			unwanted_terms={set of terms we don’t want in query}
			for each index_term in the vector:
				if index_term does NOT exist in un_wanted_terms:
					for each query_term in the query:
						similarity score = caculate similarity between index_term and query_term
						if (the similarity score>65%  AND index_item is a synonym of query_term):
							add index_item to query
						else:
							add index_item to unwanted_terms //this is important as we don’t want to keep calculating similarity results for words which we have done already
```

Level-2 simply has a slight change to it  by adding the index_term on the conditions that the similarity score be above 65% and it be a synonym. As a result, this allows it to be more lenient and during the recall scores we see slightly higher results, however we lose precision.  

Analysis for Experiment 2:

Upon analysing the P.10 and MAP scores we can see that the results slightly dip. Furhtermore, by analysing the level-2 and level-1 scores we can conclude that by the more leniante we become with expandging our queries the lower the precision scores. Conversly, we see the recal scores increase. Furthermore, this can be improved by using other potential query modifcation and expansion techniques such as eg Metric Correlation Matrix, Statistical Thesaurus etc.

