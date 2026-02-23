import click
import numpy as np
import pandas as pd
import scipy
from scipy.stats._mannwhitneyu import _broadcast_concatenate, _get_mwu_z
from scipy.stats._stats_py import _rankdata

from ao3_tags import DATA_PATH


@click.command()
@click.option('--warning', default='abuse',
              prompt='Warning for which to test annotation distributions',)
@click.option('--job_id', default='29d13398b5',
              prompt='ID of the job. Used to select files.')
def main(warning, job_id):
    dist_dir = DATA_PATH.parent / "stats" / "annotations" / warning
    df_w = pd.read_parquet(dist_dir / f"{job_id}_annotations_word.parquet")

    # Get the annotation distribution for random passages
    random_dist = df_w.loc[(df_w["pos_tag"] == "Random")]["n_pos_annotations"].iloc[0]

    # Perform a MWU-test by comparing the annotation distributions of each word against the random distribution
    df_mwu = mwu_against_random(df=df_w.loc[df_w["pos_tag"] != "Random"], random_dist=random_dist,
                                keep_cols=["word", "pos_tag", "vocabulary", "jsd", "z_two_sided_all"])
    out_file = dist_dir / f"{job_id}_mwu.csv"
    df_mwu.to_csv(out_file, index=False)
    print(f"Saved results of Mann-Whitney U-test to {out_file}")


def mwu_full(x, y):
    """
    Copied from https://github.com/scipy/scipy/blob/v1.14.1/scipy/stats/_mannwhitneyu.py
    Adapted to also report z-score and effect size
    """
    x, y, xy = _broadcast_concatenate(x, y, 0)
    n1, n2 = x.shape[-1], y.shape[-1]

    ranks, t = _rankdata(xy, 'average', return_ties=True)
    R1 = ranks[..., :n1].sum(axis=-1)
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1

    # Apply two-sided test; Get z-Score and p-value
    U, f = np.maximum(U1, U2), 2
    z = _get_mwu_z(U, n1, n2, t, continuity=True)
    p = scipy.stats.norm.sf(abs(z)) * f

    # Calculate the common language effect size
    cl_effect_size = U1 / (n1 * n2)
    return {
        "p_anno": p,
        "U1": U1,
        "U2": U2,
        "z_anno": float(z),
        "effect_size": float(cl_effect_size),
        "n1": n1,
        "n2": n2
    }


def mwu_against_random(df, random_dist, keep_cols=None, alpha=0.05):
    keep_cols = keep_cols or ["word", "pos_tag"]
    mwu_results = []
    for _, row in df.iterrows():
        dict_ = {c: row[c] for c in keep_cols}
        if len(row["n_pos_annotations"]) == 0:
            continue
        dict_.update(mwu_full(x=row["n_pos_annotations"], y=random_dist))
        mwu_results.append(dict_)

    df = pd.DataFrame(mwu_results).sort_values("effect_size", ascending=False)
    n_tests = df.shape[0]
    df["significant"] = df["p_anno"].apply(lambda p: p * n_tests <= alpha)
    return df


if __name__ == "__main__":
    main()