import numpy as np
import pandas as pd

from tqdm import tqdm

from ao3_tags import DATA_PATH


def std_effect_sizes(row: pd.Series):
    for suffix in ["", "_all"]:
        row[f"std_effect_size{suffix}"] = _std_effect_size(z=row[f"z{suffix}"], n1=row[f"n_test{suffix}"],
                                                           n2=row[f"n_baseline{suffix}"])
    return row


def _std_effect_size(z, n1, n2):
    return z / np.sqrt(n1 + n2)


def cl_effect_sizes(row: pd.Series):
    for suffix in ["", "_all"]:
        row[f"cl_effect_size{suffix}"] = _cl_effect_size(u2=row[f"u_baseline{suffix}"], n1=row[f"n_test{suffix}"],
                                                         n2=row[f"n_baseline{suffix}"])
    return row


def _cl_effect_size(u2, n1, n2):
    try:
        return u2 / (n1 * n2)
    except:
        return None


def two_sided_z_score(row: pd.Series):
    for suffix in ["", "_all"]:
        row[f"z_two_sided{suffix}"] = (
                abs(float(row[f"z{suffix}"])) * (1 if row[f"cl_effect_size{suffix}"] >= 0.5 else -1)
        )
    return row


def run(warning: str, job_id: str) -> pd.DataFrame:
    """
    :param warning:     Warning for which to calculate effect sizes
    :param job_id:      ID of the job to be run. Used to select files.
    """

    # Read the dataframe (try to use the post-processed one)
    dir_ = DATA_PATH / 'mannwhitneyu'
    z_score_path = dir_ / 'z_scores' / warning / f'{job_id}.parquet'
    processed_path = dir_ / 'effect_sizes' / warning / f'{job_id}.parquet'
    (dir_ / 'effect_sizes' / warning).mkdir(exist_ok=True)

    print("Loading dataframe...")
    df = pd.read_parquet(z_score_path)

    # Drop and convert the columns
    df = (
        df.drop(columns=[
            'r_test', 'r_baseline', 'r_test_all', 'r_baseline_all', "mu_u", "mu_u_all", "sigma_u", "sigma_u_all",
            "n1n2", "n1n2_all", "n1n2_div12", "n1n2_div12_all", "n", "n_all", "tied_rank_correction",
            "tied_rank_correction_all", "tied_rank_sum", "tied_rank_sum_all"])
        .astype({
            'z': 'float32',
            'z_all': 'float32',
            'u': 'float32',
            'u_test': 'float32',
            'u_baseline': 'float32',
            'n_test': 'int32',
            'n_baseline': 'int32',
            'u_all': 'float32',
            'u_test_all': 'float32',
            'u_baseline_all': 'float32',
            'n_test_all': 'int32',
            'n_baseline_all': 'int32',
        }))

    changed = False

    if "std_effect_size" not in df.columns:
        tqdm.pandas(desc="Calculating standardized effect sizes")
        df = df.progress_apply(lambda row: std_effect_sizes(row=row), axis=1)
        df.to_parquet(processed_path)
        changed = True

    if "cl_effect_size" not in df.columns:
        tqdm.pandas(desc="Calculating common language effect sizes")
        df = df.progress_apply(lambda row: cl_effect_sizes(row=row), axis=1)
        df.to_parquet(processed_path)
        changed = True

    if "z_two_sided" not in df.columns:
        tqdm.pandas(desc="Making z-scores two-sided")
        df = df.progress_apply(lambda row: two_sided_z_score(row=row), axis=1)
        df.to_parquet(processed_path)
        changed = True

    if changed:
        print(f"\nSaved df with p-values and effect sizes to {processed_path}\n")
    return df


if __name__ == '__main__':
    # Parse the input arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('warning', metavar='w', type=str,
                        help="Warning for which to calculate effect sizes")
    parser.add_argument('job_id', metavar='i', type=str,
                        help="ID of the job to be run. Used to select files.")
    args = parser.parse_args()

    # Run the job as specified
    _ = run(warning=args.warning, job_id=args.job_id)
