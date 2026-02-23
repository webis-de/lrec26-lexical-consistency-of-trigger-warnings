import pandas as pd
import scipy
from tqdm import tqdm

from ao3_tags import DATA_PATH
import ao3_tags.mannwhitneyu.effect_sizes as effect_sizes


def filter_z(row, min_val=1.65):
    """
    A z-score of +/- 1.65 is equivalent to a 90% confidence interval
    """
    if row["z"] and abs(row["z"]) >= min_val:
        return True
    if row["z_all"] and abs(row["z_all"]) >= min_val:
        return True
    return False


def p_values(row: pd.Series, two_sided=True):
    for suffix in ["", "_all"]:
        row[f"p_val{suffix}"] = _p_value(z=row[f"z{suffix}"], two_sided=two_sided)
    return row


def _p_value(z: float, two_sided=True):
    p = scipy.stats.norm.sf(abs(z))
    return p*2 if two_sided else p


def run(warning: str, job_id: str, z_limit=1.65, alpha=0.05):
    """
    :param warning:     Warning for which to identify significant words
    :param job_id:      ID of the job to be run. Used to select files.
    :param z_limit:     Absolute value of z to filter rows before calculating p-values
    :param alpha:       Significance level to pick significant words
    """

    # Read the dataframe (try to use the post-processed one)
    dir_ = DATA_PATH / 'mannwhitneyu'
    processed_path = dir_ / 'effect_sizes' / warning / f'{job_id}.parquet'
    significant_path = dir_ / 'significant_terms' / warning / f'{job_id}.csv'
    (dir_ / 'significant_terms' / warning ).mkdir(exist_ok=True)

    print("Loading dataframe...")
    if processed_path.is_file():
        df = pd.read_parquet(processed_path)

    else:
        df = effect_sizes.run(warning=warning, job_id=job_id)

    # Filter for significance and write the result
    tqdm.pandas(desc=f"Filtering for z-Values outside of +/-{z_limit}")
    df = df[df.progress_apply(lambda row: filter_z(row, min_val=z_limit), axis=1)]

    tqdm.pandas(desc="Calculating p-values")
    df = df.progress_apply(lambda row: p_values(row=row, two_sided=True), axis=1)
    df_sig = df.loc[(df["p_val"] <= alpha) | (df["p_val_all"] <= alpha)].sort_values("z_two_sided_all", ascending=False)
    df_sig.to_csv(significant_path, index=False)
    print(f"\nSaved df of significant terms to {significant_path}\n")


if __name__ == '__main__':
    # Parse the input arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('warning', metavar='w', type=str,
                        help="Warning for which to identify significant words")
    parser.add_argument('job_id', metavar='i', type=str,
                        help="ID of the job to be run. Used to select files.")
    args = parser.parse_args()

    # Run the job as specified
    run(warning=args.warning, job_id=args.job_id)