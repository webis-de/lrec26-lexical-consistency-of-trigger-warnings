import multiprocessing as mp
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import random
from resiliparse.extract.html2text import extract_plain_text
import spacy
from tqdm import tqdm

from ao3_tags import DATA_PATH


def main(warning, job_id, batch_size=64):
    # Ensure that the output directory exists
    output_dir = DATA_PATH / "extract" / "passages" / "passages" / warning
    os.makedirs(output_dir, exist_ok=True)
    root_path = output_dir / f"{job_id}.parquet"

    # If the output file exists, read the existing (passage) ids to filter the chapters
    processed_ids = []
    if root_path.is_dir():
        processed_ids = list(set((pq.ParquetDataset(root_path).read(columns=["id"]).to_pandas()
                                  ["id"].apply(lambda id_: "-".join(id_.split("-")[:-1]))
                                  .to_list())))
        print(f"\n\nAn existing directory was found under {root_path}. "
              f"Passages for missing chapters will be added to that directory.\n\n")

    # Read the dataframe of chapters, filtered for those that have category_tags
    print("Loading the input data...")
    matches_dataset = pq.ParquetDataset(
        DATA_PATH / "extract" / "passages" / "chapter_matches" / warning / job_id,
        filters=(~ds.field("category_tags").is_null() & ~ds.field("chapter_id").isin(processed_ids))
    )
    df_matches = matches_dataset.read().to_pandas()
    if "chapter_id" not in df_matches.columns:
        df_matches = df_matches.reset_index()

    # Create a ParquetDataset of the chapter_content to read it in batches
    content_dataset = ds.dataset(DATA_PATH / "extract" / "passages" / "chapter_content" / warning / f"{job_id}.parquet",
                                 format="parquet")
    iter_ = content_dataset.to_batches(batch_size=batch_size, filter=(~ds.field("chapter_id").isin(processed_ids)))

    # Call the function with appropriate arguments
    process_batches_in_parallel(df_matches, iter_, root_path, batch_size)


def process_batches_in_parallel(df, iter_, root_path, batch_size):
    # Function to yield arguments for batch processing
    def batch_generator():
        for batch in iter_:
            if len(batch) > 0:
                yield (df, batch)

    # Create a pool of worker processes
    with mp.Pool(processes=mp.cpu_count() // 2) as pool:
        # Process the batches using the batch generator and write results
        for result_table in tqdm(
            pool.imap_unordered(process_batch_wrapper, batch_generator()),
            total=df.shape[0] // batch_size,
            desc=f"Processing batches of {batch_size}",
        ):
            if result_table is not None:
                pq.write_to_dataset(result_table, root_path=root_path)


def process_batch_wrapper(args):
    df, batch = args
    return process_batch(df=df, batch=batch, nlp=None)


def process_batch(df: pd.DataFrame, batch: pa.lib.RecordBatch, nlp: spacy = None):
    if len(batch) == 0:
        return None

    batch_df = batch.to_pandas().merge(df, on="chapter_id")
    if batch_df.empty:
        return None

    nlp = nlp or spacy.load('en_core_web_sm')

    if "chapter_id" not in batch_df.columns:
        batch_df = batch_df.reset_index()
    results = batch_df.apply(lambda row: process_row(row=row, nlp=nlp), axis=1)
    if len(results) > 0:
        output_df = pd.concat(results.values, ignore_index=True)
        return pa.Table.from_pandas(output_df)
    else:
        return None


def process_row(row: pd.Series, nlp: spacy, n_neighbours: int = 2, n_random: int = 3):
    # 1. Remove any HTML from the chapter text and tokenize it; Turn the matches into a list of tuples
    chapter_text = _remove_html(row["chap_content"])
    doc = nlp(chapter_text)
    chapter_matches = [tuple(t) for t in row["matches"]]
    sentences = list(doc.sents)

    # 2. Get the indices of all matching words (chapter_matches)
    idx_dict = _get_passage_indices(doc=doc, chapter_matches=chapter_matches, n_neighbours=n_neighbours)
    idx_dict = _merge_passage_indices(idx_dict)

    # 3. Add random passages
    n_sentences = len(sentences)
    min_sentences = len(idx_dict) * 2 + n_random*(n_neighbours*2+1)
    if n_sentences > min_sentences:
        selected_rand, tries = 0, 0
        while selected_rand < n_random:
            rand_idx = random.randint(n_neighbours + 1, len(sentences) - n_neighbours - 1)
            rand_tuple = (rand_idx - n_neighbours, rand_idx + n_neighbours + 1)
            if rand_tuple not in idx_dict:
                idx_dict[rand_tuple] = [("Random", "Random")]
                selected_rand += 1
            tries += 1
            if tries >= n_random*5:
                break

    # 4. Return chapter metadata (ID and tags), passages of sentences and which words they contain
    results = [{"id": f'{row["chapter_id"]}-{i}',
                "category_tags": row["category_tags"],
                "matches": v,
                "passage": " ".join([str(s) for s in sentences[k[0]:k[1]]])
                }
               for i, (k, v) in enumerate(idx_dict.items())]
    return pd.DataFrame(results)


def _get_passage_indices(doc, chapter_matches, n_neighbours = 2):
    """
    Get a dictionary of sentence indices and the words that are contained in them.
    The output is a dictionary of the following format:
    :key:       Tuple of sentence indices (first sentence, sentence with match, last sentence + 1)
    :value:     List of tuples with words in the passage: (word, pos_tag)
    """
    # Get the sentences of the text
    sentences = list(doc.sents)
    num_sent = len(sentences)

    # Iterate over all tokens in the text and record the sentence indices of the passage if the token is among the chapter_matches
    idx_dict = {}
    for token in doc:
        search_tup = (token.lemma_.lower(), token.pos_)
        if search_tup in chapter_matches:
            # If a match was found, determine the idx of the current sentence and get the neighbouring sentences for the passage idx
            # The passage_idx is a tuple of (first sentence, sentence with match, last sentence + 1)
            target_sentence = token.sent
            target_idx = sentences.index(target_sentence)
            passage_idx = (max(target_idx - n_neighbours, 0), target_idx, min(target_idx + n_neighbours + 1, num_sent))

            # Add the search_tup (word, pos_tag) to the list for the current passage idx
            if passage_idx in idx_dict:
                idx_dict[passage_idx] += [search_tup]
            else:
                idx_dict[passage_idx] = [search_tup]
    return idx_dict


def _merge_passage_indices(idx_dict):
    """
    Merge two passage_idx if the idx for the matching sentence is also part of the following tuple
    Example: (1, 3, 6) and (2, 4, 7) are merged because sentence 3 is also part of the second tuple
    """
    idx_list = [(k, v) for k, v in idx_dict.items()]
    start = None
    current_matches = []
    merged_idx = {}

    for i in range(len(idx_list) - 1):
        passage_idx, match_list = idx_list[i]
        next_idx = idx_list[i + 1][0]
        start = start or passage_idx[0]
        current_matches += match_list

        # If the idx tuples do not overlap, save the current matches and reset the variables
        if passage_idx[1] < next_idx[0]:
            merged_idx[(start, passage_idx[2])] = current_matches
            start = None
            current_matches = []
        else:
            continue

    # Add the last match
    if len(idx_list)>0:
        passage_idx, match_list = idx_list[-1]
        start = start or passage_idx[0]
        current_matches += match_list
        merged_idx[(start, passage_idx[2])] = current_matches
    return {k: list(set(v)) for k, v in merged_idx.items()}


def _remove_html(chap_text: str) -> str:
    # Extract the plain text until no more changes are made
    new_len = 0
    while len(chap_text) != new_len:
        new_len = len(chap_text)
        chap_text = extract_plain_text(chap_text)
    return chap_text

if __name__ == '__main__':
    # Parse the input arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('warning', metavar='w', type=str,
                        help="Warning for which to find passages")
    parser.add_argument('job_id', metavar='i', type=str,
                        help="ID of the job to be run. Used to select tags")
    parser.add_argument('batch_size', metavar='b', type=int,
                        help="Number of chapters to process in one batch")
    args = parser.parse_args()

    # Run the job as specified
    main(warning=args.warning, job_id=args.job_id, batch_size=args.batch_size)