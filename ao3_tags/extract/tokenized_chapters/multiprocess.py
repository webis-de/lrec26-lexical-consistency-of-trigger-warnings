import json
import math
import multiprocessing as mp
import nltk
import os
from pandas import read_csv
import spacy
from tqdm import tqdm
import time
from typing import Dict, Set

from ao3_tags import DATA_PATH, RESOURCE_PATH
from ao3_tags.utils.queries import (create_es_input, generate_query, get_chapter_tags, get_number_of_shards,
                                    API_KEY, CLIENT)
from ao3_tags.utils.tokenization import tokenize_chapter
from ao3_tags.utils.wordlists import prepare_stopwords

QUERY_TYPES = {
    "warnings_fine_open": {"should_column": "warnings_fine_open"},
    "freeform": {"should_column": "tag"}
}


def process_hit(hit: Dict, nlp, output_dir: str, custom_stopwords: Set[str]):
    pid = mp.current_process().pid

    chapter_metadata = get_chapter_tags(hit)
    if chapter_metadata:
        with open(f"{output_dir}/chapters_extracted-{pid}.jsonl", "a") as file:
            json.dump(chapter_metadata, file)
            file.write("\n")

    for token_dict in tokenize_chapter(element=hit, nlp=nlp, stopwords=custom_stopwords):
        with open(f"{output_dir}/words_extracted-{pid}.jsonl", "a") as file:
            json.dump(token_dict, file)
            file.write("\n")


def es_scroll(index, search_body, scroll_duration, slice_id, processes, output_dir, custom_stopwords, progress, lock):
    # Preparation
    body = search_body.copy()
    body["slice"] = dict(id=slice_id, max=processes)
    nlp = spacy.load('en_core_web_sm')

    # Get results
    try:
        results = CLIENT.search(index=index, body=body, scroll=scroll_duration)
        sid = results['_scroll_id']
        scroll_size = len(results['hits']['hits'])
    except Exception as e:
        print("Failed to search with Exception: ", e)
        exit(1)

    while scroll_size > 0:
        # Process results
        for hit in results['hits']['hits']:
            process_hit(hit, nlp=nlp, output_dir=output_dir, custom_stopwords=custom_stopwords)

        with lock:
            progress.value += scroll_size

        # Continue scrolling
        try:
            results = CLIENT.scroll(scroll_id=sid, scroll=scroll_duration)
        except Exception as e:
            print("Failed to scroll with Exception: ", e)
            exit(1)

        sid = results['_scroll_id']
        scroll_size = len(results['hits']['hits'])

    # Call clear_scroll API after all the scrolling is done
    CLIENT.clear_scroll(body={'scroll_id': sid})


def run(warning, processes, query_type="warnings_fine_open", index='ao3-v3-chapters', size=50, scroll_duration='10m',
        max_clause_count=1_000):
    # 0.Preparation
    # Number of processes needs to be larger than 1
    assert processes > 1, print("Please provide at least two processes. "
                                "Otherwise, Elasticsearch will not accept multiprocessing.")

    # Check that the query_type is valid
    assert query_type in QUERY_TYPES, (
        print(f"Please provide a valid query type: {', '.join(list(QUERY_TYPES.keys()))}"))

    # Check if the number of processes does not exceed the number of shards
    try:
        shards = get_number_of_shards(index)
        assert processes <= shards, print(f"The number of processes ({processes}) exceeds the number of shards "
                                          f"({shards}). Please reduce the number.")
    except:
        print(f"Unable to retrieve the number of shards for index {index}. Please check that you have set a Kibana key in config.yaml."
              "If the number of processes exceeds the number of shards, the process will fail.")
        exit(1)

    # Create the directory
    output_dir = str(DATA_PATH / "extract" / "tokenized_chapters" / warning / "jsonl")
    os.makedirs(output_dir, exist_ok=True)

    # Create a list for constructing the query
    should_list = _create_should_list(warning=warning, query_type=query_type)
    if len(should_list) == 0 and query_type == "warnings_fine_open":
        print(f"\nFound no entries for {warning} in {RESOURCE_PATH / f'query_construction/{query_type}.csv'}")
        print(f"Will try to run the job with tags from {RESOURCE_PATH / 'tags' / warning}.csv")
        query_type = 'freeform'
        should_list = _create_should_list(warning=warning, query_type=query_type)

    # Ensure that the should_list is not longer than the max_clause_count
    if len(should_list) > max_clause_count:
        iter_ = math.ceil(len(should_list) / max_clause_count)
        print(f"\nNumber of query terms exceeds the maximum number of clauses. Will run {iter_} iterations.")

        for i in range(iter_):
            print(f"Iteration {i+1} of {iter_}")
            local_list = should_list[i * max_clause_count:(i + 1) * max_clause_count]
            _run_query(should_list=local_list, output_dir=output_dir, processes=processes, query_type=query_type,
                       index=index, size=size, scroll_duration=scroll_duration)
    else:
        _run_query(should_list=should_list, output_dir=output_dir, processes=processes, query_type=query_type,
                   index=index, size=size, scroll_duration=scroll_duration)


def _run_query(should_list, output_dir, processes, query_type="warnings_fine_open", index='ao3-v3-chapters', size=50,
               scroll_duration='10m'):
    query = generate_query(query_type=query_type, should_list=should_list)

    # 2. Prepare the parameters
    _, search_body = create_es_input(api_key=API_KEY, query=query)
    search_body["size"] = size
    stopwords, characters = prepare_stopwords()
    custom_stopwords = {*stopwords, *characters}

    # 3. Run the processing in parallel
    manager = mp.Manager()
    progress = manager.Value('i', 0)
    lock = manager.Lock()

    result = CLIENT.count(index=index, query=query)
    total_hits = result["count"]
    pbar = tqdm(total=total_hits, desc="Processing chapters")
    last_count = 0

    with mp.Pool(processes=processes) as pool:
        args = [(index, search_body, scroll_duration, slice_id, processes, output_dir, custom_stopwords, progress, lock)
                for slice_id in range(processes)]
        pool.starmap_async(es_scroll, args)

        # Update the progress bar according to progress.value changes
        while True:
            pbar.update(progress.value - last_count)
            last_count = progress.value
            if progress.value == total_hits:
                break
            time.sleep(0.1)

        pbar.close()


def _create_should_list(warning: str, query_type: str):
    if query_type == "warnings_fine_open":
        df = read_csv(RESOURCE_PATH / f"query_construction/{query_type}.csv")
        df = df.loc[df["warning"] == warning]
        should_list = df[QUERY_TYPES[query_type]["should_column"]].tolist()

    else:
        tag_df = read_csv(RESOURCE_PATH / "tags" / f"{warning}.csv")
        should_list = tag_df["dst"].tolist()
    return should_list


if __name__ == '__main__':
    import argparse
    # Ensure that all necessary nltk packages are available
    nltk.download('words')
    nltk.download('punkt')

    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("warning", metavar="w", type=str,
                        help='Warning for which to collect and tokenize chapters')
    parser.add_argument("--processes", metavar="p", type=int, default=8,
                        help='Number of processes for parallel processing')
    parser.add_argument("--query_type", metavar="q", type=str, default="warnings_fine_open",
                        help='Type of query to use for retrieval')
    args = parser.parse_args()

    run(warning=args.warning, query_type=args.query_type, processes=args.processes)