import os
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Sequence

from ao3_tags import DATA_PATH, RESOURCE_PATH


def get_category(job_id: str):
    job_df = pd.read_csv(DATA_PATH / "job_metadata.csv")
    capitalized_category = eval(job_df.loc[job_df["id"] == job_id]["test_categories"].iloc[0])[0]
    return capitalized_category.lower().replace(" ", "_")


def filter_df(df: pd.DataFrame, word_pos_list: Sequence[str]) -> pd.DataFrame:
    df["word_pos"] = df.apply(lambda x: f'{x["word"]}_{x["pos_tag"]}', axis=1)
    return df.loc[df["word_pos"].isin(word_pos_list)].drop("word_pos", axis=1).dropna()


def run(warning: str, job_id: str):
    """
    Create joint CSV-file with Log-Ratios and z-Scores from Mann-Whitney U-Test; Filtered by vocabulary words
    """
    # Determine the category from the job_id
    category = get_category(job_id)
    output_dir = DATA_PATH.parent / "stats" /"vocabulary" / warning
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataframes
    print("- Loading Dataframe of z-scores from Mann-Whitney U-test...")
    mwu_path = DATA_PATH / 'mannwhitneyu' / 'effect_sizes' / warning / f'{job_id}.parquet'
    mwu_df = pd.read_parquet(mwu_path)

    print("- Loading Dataframe with values for Jensen Shannon Divergence...")
    # Calculate the full JSD divergence and the share of each word
    jsd_path = DATA_PATH / 'jensen_shannon_div' / 'jsd' / warning / f'{job_id}.parquet'
    jsd_df = pd.read_parquet(jsd_path)
    jsd_df["jsd_corpus"] = jensenshannon(jsd_df["p"], jsd_df["p_baseline"])
    jsd_df["jsd_contribution"] = jsd_df["jsd"] / jsd_df["jsd_corpus"]

    print("- Loading Dataframe of log ratios...")
    lr_path = DATA_PATH / 'log_ratio' / 'log_ratios' / warning / f'{job_id}.parquet'
    lr_df = pd.read_parquet(lr_path)

    # Filter the DFs for words in the vocabulary and join them
    print("- Filtering and merging the dfs...")
    df = pd.read_csv(RESOURCE_PATH / "categories" / f"{category}_words.csv")
    word_pos_list = [f"{tup[0]}_{tup[1]}" for tup in df[["word", "pos_tag"]].itertuples(index=False, name=None)]

    mwu_df = filter_df(mwu_df, word_pos_list)
    jsd_df = filter_df(jsd_df, word_pos_list)
    lr_df = filter_df(lr_df, word_pos_list)
    df = mwu_df.merge(jsd_df[["word", "pos_tag", "jsd", "jsd_corpus", "jsd_contribution", "p", "p_baseline",
                              "n_words", "n_works"]], on=["word", "pos_tag"])
    df = df.merge(lr_df[["word", "pos_tag", "log_ratio"]], on=["word", "pos_tag"])
    df = df.sort_values(["z_two_sided_all", "jsd", "log_ratio"], ascending=[False, False, False])

    # Add p values and p values with Bonferroni corrected p-values (*2 for two-sided)
    df["p_val_all"] = df["z_two_sided_all"].apply(lambda z: stats.norm.sf(abs(z))*2)
    df["p_val_bonferroni_all"] = df["p_val_all"].apply(lambda p: min(p * df.shape[0], 1))

    # Save the result
    df.to_csv(output_dir / f'{job_id}.csv', index=False)
    print(f"\nSaved dataframe to {output_dir / f'{job_id}.csv'}")
    return df


if __name__ == "__main__":
    # Parse the input arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('warning', metavar='w', type=str,
                        help="Warning for which to join results for Mann-Whitney U-test, Jensen Shannon Divergence, and Log Ratio")
    parser.add_argument('job_id', metavar='i', type=str,
                        help="ID of the job. Used to select files.")
    args = parser.parse_args()

    # Run the job as specified
    _ = run(warning=args.warning, job_id=args.job_id)
