import re
import os
import sys
import datetime
import argparse
from query import Query
from invertedindex import InvertedIndex


QUERY_FILE = os.path.join(os.path.dirname(
    __file__), "../data/topics_MB1-49.txt")


def parse_query_file():
    raw_queries = open(QUERY_FILE).read()
    query_pattern = re.compile("<title> (.*) </title>")
    queries = re.findall(query_pattern, raw_queries)
    return queries


def run_query(raw_query, top=1000):
    query = Query(raw_query)
    query.process_raw_query()
    return query.perform_query(top=top)


def run_batch_query(queries):
    curr_run = datetime.datetime.now().strftime("%d/%m/%Y:%H:%M:%S")
    with open('Results.txt', 'w', newline='') as file:
        print(" ".join(["topic_id", "Q0", "docno",
                        "rank", "score", "tag"]), file)
        for i in range(len(queries)):
            raw_query = queries[i]
            results = run_query(raw_query)
            rank = 1
            for id, sim in results.items():
                print(" ".join([str(i + 1), "Q0", str(id),
                                str(rank), str(sim), curr_run]), file)
                rank += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-Q', '--query', help='string to query', required=False)
    parser.add_argument(
        '-F', '--file', help='Run queries defined in ../data/topics_MB1-49.txt', action='store_true', required=False)
    args = parser.parse_args()
    if args.query:
        results = run_query(args.query, top=10)
        print("\nTOP TEN RESULTS\n")
        for id, sim in results.items():
            print("ID: " + str(id) + "\n")
            print("Raw Message: " +
                  str(Query.get_doc_index().raw_db[id]) + "\n")
            print("Similarity: " + str(sim) + "\n\n")
    elif args.file:
        queries = parse_query_file()
        run_batch_query(queries)


if __name__ == "__main__":
    main()
