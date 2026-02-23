import os

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from ao3_tags import DATA_PATH


def sample_passages(warning: str, job_id: str, num_passages: int = 10_000, min_len: int = 500, max_len: int = 1_000):
    """
    Take a parquet file of passages for a job (created by ao3_tags/extract) and sample passages for annotation
    """
    # Assert correctness of filtering conditions
    assert max_len > min_len, print("The max_len needs to be larger than the min_len")

    # Load the dataframe of all passages for the defined job; Set sampled to 0
    dataset = pq.ParquetDataset(DATA_PATH / "extract" / "passages" / "passages" / warning / f"{job_id}.parquet")
    df = dataset.read(columns=["id", "matches", "passage"]).to_pandas()
    df["sampled"] = 0

    # Load the dataframe of words and get the tuples of (word, pos_tag)
    word_df = pd.read_csv(DATA_PATH / "extract" / "passages" / "sampled_words" / warning / f"{job_id}.csv")
    word_tuples = [(w, pos_tag) for w, pos_tag in zip(word_df["word"], word_df["pos_tag"])]

    passages_per_word = num_passages // len(word_tuples)

    # Filter the dataframe using min_len and max_len; Turn the matches in tuples (instead of arrays)
    df = df.loc[df["passage"].apply(lambda passage: min_len <= len(passage) <= max_len)]
    df["matches"] = df["matches"].apply(lambda matches: [tuple(x) for x in matches])

    # Sample passages uniformly for each word
    sampled_df = pd.DataFrame()
    for word_tup in tqdm(word_tuples, desc="Sampling passages uniformly for each word tuple"):
        df, sampled_df = _sample_for_tup(word_tup=word_tup, df=df, sampled_df=sampled_df,
                                         num_per_tup=passages_per_word)

    # Fill any missing passages after uniform word sampling;
    # Weighted sampling based on frequency of each combination of words (word_str) in the passages
    df["word_str"] = df["matches"].apply(lambda x: ",".join([f"{t[0]}_{t[1]}" for t in sorted(x)]))
    unsampled_df = df.loc[(df["sampled"] == 0) &
                          (~df["matches"].apply(lambda matches: ("Random", "Random") in matches))].copy(deep=True)
    unsampled_df["weight"] = unsampled_df.groupby("word_str")["word_str"].transform("count").apply(lambda x: 1 / x)
    add_samples = unsampled_df.sample(num_passages - len(sampled_df), weights="weight")[["id", "matches", "passage"]]
    sampled_df = pd.concat([sampled_df, add_samples], ignore_index=True).sort_values("id")

    # Sample additional random passages (10% of num_passages)
    print("Sampling random passages...")
    _, sampled_df = _sample_for_tup(word_tup=("Random", "Random"), df=df, sampled_df=sampled_df,
                                    num_per_tup=num_passages // 10)
    # Return the result
    return sampled_df


def _sample_for_tup(word_tup: tuple, df: pd.DataFrame, sampled_df: pd.DataFrame, num_per_tup: int):
    """
    Sample passages for a word tuple. Preferably, the passages contain only the current word
    """
    # Select unsampled passages that contain the current word
    w_df = df.loc[
        (df["sampled"] == 0) &
        (df["matches"].apply(lambda matches: word_tup in matches))
        ]

    # Reduce the w_df to passages that only contain the current word to reduce noise; Sample from there
    exclusive_w_df = w_df.loc[(w_df["matches"].apply(lambda matches: word_tup == matches))]
    w_sample = exclusive_w_df.sample(min(num_per_tup, len(exclusive_w_df)))

    # If insufficient exclusive matches were found, add missing rows from the w_df
    num_sampled = w_sample.shape[0]
    if num_sampled < num_per_tup:
        add_w_df = w_df.loc[~w_df["id"].isin(w_sample["id"])]
        add_sample = add_w_df.sample(min(num_per_tup - num_sampled, len(add_w_df)))
        w_sample = pd.concat([w_sample, add_sample])

    # Add the sample to the sampled_df
    sampled_df = pd.concat([sampled_df, w_sample[["id", "matches", "passage"]]], ignore_index=True)

    # Indicate which passages were sampled
    df.loc[df["id"].isin(w_sample["id"].to_list()), "sampled"] = 1
    return df, sampled_df



def load_sampled_passages(warning: str, job_id: str, num_passages: int = 10_000, min_len: int = 500,
                          max_len: int = 1_000):
    """
    :param warning:             Warning for which to load sampled passages
    :param job_id:              ID of the job to run. Used to define the file
    :param num_passages:        Number of passages to sample if no ID file is found
    :param min_len:             Minimum number of characters for a passage to be sampled
    :param max_len:             Maximum number of characters for a passage to be sampled
    """
    passage_path = DATA_PATH / "extract" / "passages" / "passages" / warning / f"{job_id}.parquet"
    id_path = DATA_PATH / "annotations" / warning / f"{job_id}_ids.txt"
    os.makedirs(id_path.parent, exist_ok=True)

    if id_path.is_file():
        print(f"- Loading passages with IDs in {id_path}")
        with open(id_path, "r") as f:
            ids = [i.replace("\n", "") for i in f.readlines()]
        dataset = pq.ParquetDataset(passage_path, filters=[("id", "in", ids)])
        return dataset.read().to_pandas().sort_values("id")

    else:
        print(f"- Sampling new passages as no IDs were found in {id_path}")
        df = sample_passages(warning=warning, job_id=job_id, num_passages=num_passages,
                             min_len=min_len, max_len=max_len)
        df = df.sort_values("id")
        with open(id_path, "w") as f:
            for i in df["id"].values:
                f.write(f"{i}\n")
        return df


def load_passages(warning: str, job_id: str):
    """
    :param warning:             Warning for which to load sampled passages
    :param job_id:              ID of the job to run. Used to define the file
    """
    passage_path = DATA_PATH / "extract" / "passages" / "passages" / warning / f"{job_id}.parquet"
    try:
        df = pd.read_parquet(passage_path)
    except:
        raise ValueError(f"{passage_path} does not exist. Please place a parquet file with passages there.")
    return df
