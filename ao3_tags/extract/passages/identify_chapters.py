import os
import pandas as pd
import pyarrow.parquet as pq
import json
from tqdm import tqdm

from ao3_tags import DATA_PATH


def main(warning, job_id):
    # Get the folder suffixes that exist for both words and chapters
    tokenization_path = DATA_PATH / "extract" / "tokenized_chapters" / warning
    word_parquet_suffixes = [p.name.replace("words-", "").replace(".parquet", "")
                             for p in tokenization_path.glob("words*.parquet")]
    chapter_parquet_suffixes = [p.name.replace("chapters-", "").replace(".parquet", "")
                                for p in tokenization_path.glob("chapters*.parquet")]
    suffixes = [s for s in word_parquet_suffixes if s in chapter_parquet_suffixes]

    # Get the word filters and category tags for the current warning and job id
    word_filters = _get_word_filters(warning, job_id)
    category_tags = _get_category_tags(job_id)

    # Ensure that the output directory exists
    output_dir = DATA_PATH / "extract" / "passages" / "chapter_matches" / warning / job_id
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the suffixes and construct the Dataframe of chapter_id, matches, category_tags
    tqdm.pandas()
    for i, suffix_num in enumerate(suffixes):
        print("\n" + "-" * 50 + "\n" + f"Suffix {i + 1}: {suffix_num}")
        suffix_df = process_suffix(tokenization_path=tokenization_path,
                                   suffix_num=suffix_num,
                                   word_filters=word_filters,
                                   category_tags=category_tags)
        suffix_df.to_parquet(output_dir / f"chapters-{suffix_num}.parquet")


def process_suffix(tokenization_path, suffix_num, word_filters, category_tags):
    word_df = load_word_df(tokenization_path, suffix_num, word_filters)
    chapter_df = load_chapter_df(tokenization_path, suffix_num, category_tags)
    print("Merging matches and chapter tags...")
    return word_df.merge(chapter_df, left_index=True, right_index=True)


def load_chapter_df(tokenization_path, suffix_num, category_tags):
    print("Identifying category tags in chapters...")
    chapter_df = pd.read_parquet(tokenization_path / f"chapters-{suffix_num}.parquet")

    def _find_category_tags(tags):
        matches = [t for t in category_tags if t in tags]
        return matches if len(matches) > 0 else None

    chapter_df["category_tags"] = chapter_df["tags"].progress_apply(lambda tags: _find_category_tags(tags))
    return chapter_df[["chapter_id", "category_tags"]].set_index("chapter_id")


def load_word_df(tokenization_path, suffix_num, word_filters):
    print("Loading dataset of word matches...")
    dataset = pq.ParquetDataset(
        tokenization_path / f"words-{suffix_num}.parquet",
        filters=word_filters
    )
    word_df = dataset.read(columns=["chapter_id", "word", "pos_tag"]).to_pandas()
    print("Aggregating matches by chapter...")
    word_df = word_df.groupby("chapter_id").agg({
        "word": list,
        "pos_tag": list
    })
    word_df["matches"] = word_df.progress_apply(
        lambda row: [(w, pos_tag) for w, pos_tag in zip(row["word"], row["pos_tag"])],
        axis=1
    )
    return word_df.drop(["word", "pos_tag"], axis=1)


# Helper functions (Load words for retrieval and category tags)
def _get_word_filters(warning: str, job_id: str):
    words_df = pd.read_csv(DATA_PATH / "extract" / "passages" / "sampled_words" / warning / f"{job_id}.csv")
    return [
        [("word", "=", x[0]), ("pos_tag", "=", x[1])]
        for x in list(words_df[["word", "pos_tag"]].itertuples(index=False, name=None))
    ]


def _get_category_tags(job_id: str):
    job_json = DATA_PATH / "job_tags.json"
    with open(job_json, "r") as file:
        job_tags = json.load(file)
        return job_tags[job_id]["test_tags"]


if __name__ == "__main__":
    import argparse

    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("warning", metavar="w", type=str, default="abuse",
                        help='Warning for which to identify chapters that contain sampled words.')
    parser.add_argument("job_id", metavar="i", type=str, default="29d13398b5",
                        help='ID of the job. Sets warning category.')
    args = parser.parse_args()
    main(warning=args.warning, job_id=args.job_id)