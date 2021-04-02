import re
import os
import sys
import datetime
import argparse
from query import Query
from inverted_index import InvertedIndex
from utils.model_utils import get_inverted_index

QUERY_FILE = os.path.join(os.path.dirname(
    __file__), "../data/topics_MB1-49.txt")


def parse_query_file():
    """
    Parses query file defined in ../data/topics_MB1-49.txt
    Returns a list of query strings
    """
    raw_queries = open(QUERY_FILE).read()
    query_pattern = re.compile("<title> (.*) </title>")
    queries = re.findall(query_pattern, raw_queries)
    return queries


def run_query(raw_query='', method='default', top=1000):
    """
    Performs a single query.
    Returns an ordered mapping of results(doc_id, similarity)

    Keyword arguments:
    query -- query to be performed(default '') 
    method -- method of query where default is tf-idf cosine similarity (default default)
    top -- top results to fetch (default 1000)
    """
    query = Query(raw_query)
    query.process_raw_query()
    return query.perform_query(method=method, top=top)


def run_batch_query(queries=[], method='default'):
    """
    Performs a list of queries, results are saved in ./Results.txt. 
    Returns void

    Keyword arguments:
    queries -- list of query strings (default [])
    method -- method of query where default is tf-idf cosine similarity (default default)
    """
    curr_run = datetime.datetime.now().strftime("%d/%m/%Y:%H:%M:%S")
    with open('Results.txt', 'w', newline='') as file:
        print(" ".join(["topic_id", "Q0", "docno",
                        "rank", "score", "tag"]), file=file)
        for i in range(len(queries)):
            raw_query = queries[i]
            results = run_query(raw_query, method=method)
            rank = 1
            for id, sim in results.items():
                print(" ".join([str(i + 1), "Q0", str(id),
                                str(rank), str(sim), curr_run]), file=file)
                rank += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-Q', '--query', help='string to query', required=False)
    parser.add_argument(
        '-F', '--file', help='Run queries defined in ../data/topics_MB1-49.txt', action='store_true', required=False)
    parser.add_argument(
        '-M', '--method', help='Neural Method for Retrieval', default='default', choices={'default', 'BERT'}, required=False)
    args = parser.parse_args()
    if args.query:
        results = run_query(args.query, method=args.method, top=10)
        print("\nTOP TEN RESULTS\n")
        for id, sim in results.items():
            print("ID: " + str(id) + "\n")
            print("Raw Message: " +
                  str(get_inverted_index().raw_db[id]) + "\n")
            print("Similarity: " + str(sim) + "\n\n")
    elif args.file:
        queries = parse_query_file()
        run_batch_query(queries, args.method)


if __name__ == "__main__":
    main()
