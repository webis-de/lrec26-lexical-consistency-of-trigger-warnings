import click
import numpy as np
import os
import pandas as pd

from ao3_tags import DATA_PATH
from ao3_tags.annotate.load_passages import load_sampled_passages
from ao3_tags.stats.annotate.utils import load_annotations


@click.command()
@click.option('--warning', default='abuse',
              prompt='Warning for which to create annotation distributions',)
@click.option('--job_id', default='29d13398b5',
              prompt='ID of the job. Used to select files.')
def main(warning, job_id):
    """
    Create parquet files that contain the distribution of positive annotations per word, word group,
    and quantiles of z-score and contribution to Jensen Shannon Divergence.
    """
    # 1. Load the input
    output_dir = DATA_PATH.parent / "stats" / "annotations" / warning
    os.makedirs(output_dir, exist_ok=True)
    df_w, df_a = load_input(warning, job_id)

    # 2. Calculate the number of positive annotations per passage
    anno_dict = df_a.groupby("passage_id")["response"].sum().apply(lambda num: int(num)).to_dict()

    # 3. Get the distribution of positive annotations per word, group, and quantile
    create_file(df_w=df_w, anno_dict=anno_dict,
                file_path=output_dir / f"{job_id}_annotations_word.parquet", group_cols=None)
    create_file(df_w=df_w, anno_dict=anno_dict,
                file_path=output_dir / f"{job_id}_annotations_group.parquet",
                group_cols=["group", "vocabulary"])
    create_file(df_w=df_w, anno_dict=anno_dict,
                file_path=output_dir / f"{job_id}_annotations_z_quantile.parquet",
                group_cols=["z_quantile", "vocabulary"])
    create_file(df_w=df_w, anno_dict=anno_dict,
                file_path=output_dir / f"{job_id}_annotations_jsd_quantile.parquet",
                group_cols=["jsd_quantile", "vocabulary"])
    create_file(df_w=df_w, anno_dict=anno_dict,
                file_path=output_dir / f"{job_id}_annotations_joined_quantiles.parquet",
                group_cols=["z_quantile", "jsd_quantile", "vocabulary"])


def create_file(df_w, anno_dict, file_path, group_cols = None):
    # If grouping columns were provided, adapt the df_w
    if group_cols is not None:
        df_w["passage_ids"] = df_w["passage_ids"].apply(lambda p_list: list(p_list))
        df_w = _passage_ids_per_group(df_w=df_w.copy(deep=True), group_cols=group_cols)

    # Save the dataframe
    df = _dist_of_pos_annotations(df=df_w, anno_dict=anno_dict, keep_cols=group_cols)
    df.to_parquet(file_path, index=False)
    print(f"Saved {file_path}")


def load_input(warning, job_id, n_quantiles: int = 4):
    """
    Load the (prepared) dataframe of words and annotations. If the word_df is not available, construct it
    """
    output_dir = DATA_PATH.parent / "stats" / "annotations" / warning
    word_df_path = output_dir / f"{job_id}_words.parquet"
    df_a = load_annotations(warning, job_id)
    if word_df_path.is_file():
        print(f"Existing file found under {word_df_path}. This will be used as basis for all operations.")
        df_w = pd.read_parquet(word_df_path)

    # If no passages_per_word.parquet is available, construct it from annotations (df_a), passages (df_p), and words (df_w)
    else:
        df_p = load_sampled_passages(warning, job_id)
        df_p["matches"] = df_p["matches"].apply(lambda m: [tuple(x) for x in m])
        df_w = pd.read_csv(DATA_PATH / "extract" / "passages" / "sampled_words" / warning / f"{job_id}.csv")
        df_w.loc[df_w.shape[0]] = {"word": "Random", "pos_tag": "Random", "group": "random", "vocabulary": -1}

        # Assign passage IDs to each word; Assign words to quantiles based on their measures; Save the result
        df_w = df_w.apply(lambda row: _assign_passage_ids(row=row, df_p=df_p), axis=1)
        df_w['z_quantile'] = pd.qcut(df_w['z_two_sided_all'], q=n_quantiles, labels=False)
        df_w['jsd_quantile'] = pd.qcut(df_w['jsd'], q=n_quantiles, labels=False)
        df_w.to_parquet(word_df_path)
    return df_w, df_a


# Helper functions
def _passage_ids_per_group(df_w, group_cols):
    """
    Group the dataframe of words by group_cols and determine the unique passage ids for the group.
    """
    tmp_df = df_w.groupby(group_cols).agg({
        "passage_ids": "sum"
    }).reset_index()
    tmp_df["passage_ids"] = tmp_df["passage_ids"].apply(lambda p_list: list(set(p_list)))
    return tmp_df


def _dist_of_pos_annotations(df, anno_dict, keep_cols=None):
    """
    Take the passage ids per df row to find the corresponding distribution of positive annotations.
    """
    keep_cols = keep_cols or ["word", "pos_tag", "vocabulary", "jsd", "z_two_sided_all"]

    # Create a copy of the df to avoid overwriting input dfs
    df = df.copy(deep=True)
    if all([c in df.columns for c in keep_cols]):
        df = df[[*keep_cols, "passage_ids"]]

    # Get the number of positive annotations; Get the mean; Drop the passage ids
    df["n_pos_annotations"] = df["passage_ids"].apply(lambda p_list: [anno_dict[id_] for id_ in p_list])
    df["mean_n_pos"] = df["n_pos_annotations"].apply(lambda a_list: float(np.mean(a_list)))
    df["n_passages"] = df["n_pos_annotations"].apply(lambda a_list: len(a_list))
    return df.drop(columns=["passage_ids"])


def _assign_passage_ids(row, df_p):
    tmp_df = df_p.loc[df_p["matches"].apply(lambda m: (row["word"], row["pos_tag"]) in m)]
    row["passage_ids"] = tmp_df["id"].to_list()
    return row


def _create_prompt_dict(prompt, df_a):
    tmp_df = df_a[["passage_id", prompt]].groupby("passage_id")[prompt].sum()
    return tmp_df.apply(lambda num: int(num)).to_dict()


if __name__ == "__main__":
    main()