import os
import pandas as pd

from ao3_tags import DATA_PATH


def main(warning: str, job_id: str, k: int = 25, min_n: int = 100):
    """
    Sample words for passage retrieval in 4 groups (Sets of 2 sampled from the vocabulary and non-vocabulary terms)
    1. Set: Top k words with the highest z-scores
    2. Set: Top k words with the highest Jensen Shannon Divergence
    Group 3 & 4: Repetition of the ones above for non-vocabulary terms
    """
    output_dir = DATA_PATH / "extract" / "passages" / "sampled_words" / warning
    os.makedirs(output_dir, exist_ok=True)

    # Load the necessary dataframes
    print("Loading dataframes...")
    vocabulary_path = DATA_PATH.parent / "stats" / "vocabulary" / warning / f'{job_id}.csv'
    voc_df = pd.read_csv(vocabulary_path)
    voc_df["word_pos"] = voc_df.apply(lambda x: f"{x['word']}_{x['pos_tag']}", axis=1)

    mwu_path = DATA_PATH / 'mannwhitneyu' / 'significant_terms' / warning / f'{job_id}.csv'
    mwu_df = pd.read_csv(mwu_path)
    mwu_df["word_pos"] = mwu_df.apply(lambda x: f"{x['word']}_{x['pos_tag']}", axis=1)
    jsd_path = DATA_PATH / 'jensen_shannon_div' / 'jsd' / warning / f'{job_id}.parquet'
    jsd_df = pd.read_parquet(jsd_path)
    jsd_df["word_pos"] = jsd_df.apply(lambda x: f"{x['word']}_{x['pos_tag']}", axis=1)
    df = pd.merge(mwu_df[["word", "pos_tag", "word_pos", "z_two_sided_all", "n_test"]],
                  jsd_df[["word_pos", "jsd", "p", "p_baseline"]],
                  on="word_pos", how="left")

    # Vocabulary Terms
    df1, df2 = create_dfs(voc_df, min_n=min_n, k=k)
    for tmp in [df1, df2]:
        tmp["vocabulary"] = 1

    # Non-Vocabulary Terms
    non_voc_df = df.loc[
        (~df["word_pos"].isin(df1["word_pos"])) &
        (~df["word_pos"].isin(df2["word_pos"]))
        ].copy(deep=True)
    df3, df4 = create_dfs(non_voc_df, min_n=min_n, k=k)
    for tmp in [df3, df4]:
        tmp["vocabulary"] = 0

    # Concat dataframes and save the words
    df_words = pd.concat([df.drop("word_pos", axis=1) for df in [df1, df2, df3, df4]])
    df_words.to_csv(output_dir / f"{job_id}.csv", index=False)
    print(f"\nSaved dataframe to {output_dir / f'{job_id}.csv'}")
    return df_words


def create_dfs(df: pd.DataFrame, k: int = 20, min_n: int = 500):
    if min_n:
        df = df.loc[df["n_test"] >= min_n]

    # Group 1: Highest z-Score
    df1 = df.sort_values("z_two_sided_all", ascending=False).iloc[:k]
    df1["group"] = "significant"

    # Group 2: Highest JSD with p > p_baseline (as JSD is symmetric) and not in df1
    df2 = df.loc[(~df["word_pos"].isin(df1["word_pos"])) &
                 (df["p"] > df["p_baseline"])].copy(deep=True)
    df2 = df2.sort_values("jsd", ascending=False).iloc[:k]
    df2["group"] = "Highest JSD"

    return (df1[["word", "pos_tag", "z_two_sided_all", "jsd", "n_test", "word_pos", "group"]],
            df2[["word", "pos_tag", "z_two_sided_all", "jsd", "n_test", "word_pos", "group"]])


if __name__ == '__main__':
    # Parse the input arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('warning', metavar='w', type=str,
                        help="'Warning for which to sample words for passage retrieval")
    parser.add_argument('job_id', metavar='i', type=str,
                        help="ID of the job. Used to select files.")
    args = parser.parse_args()

    # Run the job as specified
    main(warning=args.warning, job_id=args.job_id)
