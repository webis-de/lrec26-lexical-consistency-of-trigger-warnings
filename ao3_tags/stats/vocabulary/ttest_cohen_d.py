import math
import pandas as pd
from scipy.stats.mstats import ttest_1samp

from ao3_tags import DATA_PATH


def t_test_and_effect_size(df: pd.DataFrame, job_id: str, col: str = "z_two_sided_all"):
    t_result = ttest_1samp(df[col], 0)
    mean = df[col].mean()
    std = df[col].std()
    cohens_d = (mean - 0) / (math.sqrt((std ** 2 + 1 ** 2) / 2))

    return {
        "job_id": job_id,
        "col": col,
        "p_value": t_result.pvalue,
        "statistic": t_result.statistic,
        "mean": mean,
        "std": std,
        "cohens_d": cohens_d,
        "sample_size": len(df)
    }


def run(warning: str):
    # Load the CSV of vocabulary words with z-score and jsd
    output_dir = DATA_PATH.parent / "stats" / "vocabulary" / warning
    file_names = [x for x in output_dir.glob("*.csv")]
    if len(file_names) == 0:
        print(f"No files found in  {output_dir}. Please run join_lr_mwu.py first.")
        exit(1)

    # Perform t-test and calculate cohen's d
    records = []
    for file_name in file_names:
        if file_name.name in ["ttest_cohens_d.csv", "base_category_effects.csv", "base_category_correlation.csv"]:
            continue

        df = pd.read_csv(file_name)
        job_id = file_name.name.replace(".csv", "")
        for col in ["jsd"]:
            records.append(t_test_and_effect_size(df=df, job_id=job_id, col=col))

    # Save the result
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "ttest_cohens_d.csv", index=False)
    print(f"\nSaved results to {output_dir}/ttest_cohens_d.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("warning", metavar="w", type=str,
                        help='Warning for which to perform the t-test and calculate effect sizes')
    args = parser.parse_args()
    run(args.warning)