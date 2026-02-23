from decimal import Decimal
import itertools
from scipy.stats import pearsonr
import pandas as pd

from ao3_tags import DATA_PATH


def main(warning: str = "abuse"):
    # Load and prepare the CSV files containing the results of the MWU-test
    dist_dir = DATA_PATH.parent / "stats" / "annotations" / warning
    mwu_dfs = {}
    for job_id, category in [("29d13398b5", "Emotional Abuse"),
                             ("e677272ffe", "Physical Abuse"),
                             ("42cf7cb406", "Sexual Abuse")]:
        df = pd.read_csv(dist_dir / f"{job_id}_mwu.csv")
        df["label"] = df["vocabulary"].apply(lambda v: "Voc." if v==1 else "N.-Voc.")
        mwu_dfs[category] = df.rename(columns={"z_two_sided_all": "z", "jsd": "jsd"})

    # Create a dataframe of correlations between the measures and the effect size on the mwu test
    out_file = dist_dir / f"pearson_r_mwu_effect_size.csv"
    df = create_corr_dfs(mwu_dfs)
    df.to_csv(out_file, index=False)
    print(f"Saved results of correlations on Mann-Whitney U-test to {out_file}")
    print("\nTables as latex code:")
    print_df_latex(df=df)


def create_corr_dfs(mwu_dfs: dict):
    # Get Pearson r and corresponding p-value for every category
    df = pd.DataFrame()
    for category, mwu_df in mwu_dfs.items():
        corr_df = _create_corr_df(mwu_df)
        corr_df["category"] = category
        df = pd.concat([df, corr_df], ignore_index=True)

    # Reindex the dataframe to get measures separated by voc and non-voc.
    voc_cols = ["voc", "non_voc"]
    df["vocabulary"] = df["vocabulary"].apply(lambda v: voc_cols[0] if v == 1 else voc_cols[1])
    idx_order = [tup for tup in itertools.product(voc_cols, ["r", "p", "n"])]
    return _reindex_corr_df(df, idx_order)


def print_df_latex(df, cat_len: int = 3, include_n: bool = False):
    """
    Print the dataframe of correlation results in latex format
    :param df:          DataFrame of correlation results
    :param cat_len:     Number of characters for the category name
    :param include_n:   Whether to include the sample size
    """
    # Format the numbers
    format_dict = {"r": '%.2f', "p": '%.1e', "n": "%.0f"}
    for v, f in itertools.product(["voc", "non_voc"], format_dict.keys()):
        df[f"{v}_{f}"] = df[f"{v}_{f}"].apply(lambda x: format_dict[f] % Decimal(x))

    # Print the latex code
    drop_cols = ["measure"] if include_n else ["measure", "voc_n","non_voc_n"]
    for measure in ["z", "jsd"]:
        tmp = df.loc[df["measure"] == measure].drop(drop_cols, axis=1)
        tmp["category"] = tmp["category"].apply(lambda c: c[:cat_len] + ".")
        print(tmp.to_latex(index=False))


def _create_corr_df(df):
    corr_df = pd.DataFrame()
    for v in df["vocabulary"].unique():
        tmp_df = df.loc[df["vocabulary"] == v][["jsd", "z", "effect_size"]]
        pearsonr_df, pvalue_df = _pearsonr_with_p(tmp_df)

        v_corr_df = pd.DataFrame({"vocabulary": v,
                                  "r": pearsonr_df["effect_size"][["z", "jsd"]],
                                  "p": pvalue_df["effect_size"][["z", "jsd"]],
                                  "n": tmp_df.shape[0]
                                  }).reset_index()
        corr_df = pd.concat([corr_df, v_corr_df], ignore_index=True)
    return corr_df.rename(columns={"index": "measure"})


def _pearsonr_with_p(df):
    # Create empty dataframes
    dfcols = pd.DataFrame(columns=df.columns)
    pearsonr_df = dfcols.transpose().join(dfcols, how='outer')
    pvalue_df = dfcols.transpose().join(dfcols, how='outer')

    # Iterate over the columns to calculate pearson r and the corresponding p-value
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            result = pearsonr(tmp[r], tmp[c])
            pearsonr_df.loc[r, c] = result[0]
            pvalue_df.loc[r, c] = result[1]
    return pearsonr_df, pvalue_df


def _reindex_corr_df(df, idx_order):
    final_df = pd.DataFrame()
    for measure in ["z", "jsd"]:
        tmp_df = df.loc[df["measure"] == measure].pivot_table(index=["category"],
                                                              columns=["vocabulary"],
                                                              values=["r", "p", "n"]
                                                              )
        tmp_df.columns = tmp_df.columns.swaplevel(0, 1)
        tmp_df = tmp_df.loc[:, idx_order].reset_index()
        tmp_df["measure"] = measure
        final_df = pd.concat([final_df, tmp_df], ignore_index=True)

    # Collapse the multiindex columns into one level index
    return final_df.set_axis(['_'.join(c) if c[1] else c[0] for c in final_df.columns], axis='columns')


if __name__ == "__main__":
    main()