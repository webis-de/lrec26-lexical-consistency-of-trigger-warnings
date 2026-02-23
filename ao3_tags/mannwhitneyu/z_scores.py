# Approach based on this paper: https://academic.oup.com/dsh/article-abstract/31/2/374/2462752
# - Traditional bag-of-words methods (log-likelihood and Chi-Square) overestimate the significance by assuming
#   independence between word frequencies (https://varieng.helsinki.fi/series/volumes/19/saily_suomela/)
# - The paper show that other methods (e.g. Mann-Whitney-U) are more strict and thus perform better
#   They assume that words across texts in two corpora are independent but not within the same text
# - Preprint available via https://users.ics.aalto.fi/lijffijt/articles/lijffijt2015a.pdf

from distutils.util import strtobool
from functools import reduce
import json
import os
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, DecimalType, FloatType, StructType, StructField
from pyspark.sql.window import Window
from scipy.stats import mannwhitneyu
from typing import Sequence


def mann_whitney_u(x, y):
    x, y = list(x), list(y)
    U, p = mannwhitneyu(x, y)
    return float(U), float(p)


mann_whitney_u_udf = f.udf(
    mann_whitney_u,
    StructType([
        StructField("U", FloatType(), False),
        StructField("p", FloatType(), False)
    ])
)


def add_zero_freq(arr: Sequence, missing_chapters: int):
    return arr + [0.0] * missing_chapters


add_zero_freq_udf = f.udf(
    add_zero_freq,
    ArrayType(FloatType())
)

# Decimal types for DataFrame columns; MAX_DIGITS_N is the number of digits for the expected maximum number of chapters
MAX_DIGITS_N = 12
SAMPLE_TYPE = DecimalType(precision=MAX_DIGITS_N, scale=0)  # Can only be integer
AVG_RANK_TYPE = DecimalType(precision=MAX_DIGITS_N, scale=1)  # max avg: (max_samples + 1) / 2; 1st decimal in [0, 5]
RANK_SUM_TYPE = DecimalType(precision=MAX_DIGITS_N * 2, scale=1)  # max is max_samples * max avg; 1st decimal in [0, 5]
SAMPLES_SQ_TYPE = DecimalType(precision=MAX_DIGITS_N * 2, scale=0)
MAX_DECIMAL_TYPE = DecimalType(precision=38, scale=16)


def increase_rank_sum(rank_sum_df: DataFrame):
    # Prepare the dataframe
    rank_sum_df = (
        rank_sum_df
        .withColumn("missing_total",
                    f.try_add(
                        f.col("missing_n_test").cast(SAMPLE_TYPE),
                        f.col("missing_n_baseline").cast(SAMPLE_TYPE))
                    .cast(SAMPLE_TYPE)
                    )
        .withColumn("avg_rank", f.try_divide(
            f.try_add(
                f.col("missing_total").cast(SAMPLE_TYPE),
                f.lit(1)).cast(SAMPLE_TYPE),
            f.lit(2).cast(SAMPLE_TYPE)
        ).cast(AVG_RANK_TYPE)
                    )
    )

    # Increase the rank sums (Skip rows that have no missing chapters by using when, otherwise)
    rank_sum_df = _rank_sum_helper(rank_sum_df, corpus="test")
    rank_sum_df = _rank_sum_helper(rank_sum_df, corpus="baseline")

    # Increase the tied rank correction for all missing documents
    rank_sum_df = (rank_sum_df
                   .withColumn("tied_rank_sum_all",
                               f.try_add(
                                   f.col("tied_rank_sum").cast(MAX_DECIMAL_TYPE),
                                   f.try_subtract(
                                       f.pow(
                                           f.col("missing_total").cast(MAX_DECIMAL_TYPE),
                                           f.lit(3).cast(MAX_DECIMAL_TYPE)
                                       ).cast(MAX_DECIMAL_TYPE),
                                       f.col("missing_total").cast(MAX_DECIMAL_TYPE)
                                   ).cast(MAX_DECIMAL_TYPE)
                               ).cast(MAX_DECIMAL_TYPE)
                               )
                   )
    return rank_sum_df


def _rank_sum_helper(rank_sum_df: DataFrame, corpus: str):
    return (
        rank_sum_df
        .withColumn(f"r_{corpus}_all", f.when(
            f.col("missing_total") == 0,
            f.col(f"r_{corpus}").cast(RANK_SUM_TYPE)
        ).otherwise(
            f.try_add(
                f.try_add(
                    f.col(f"r_{corpus}").cast(RANK_SUM_TYPE),
                    f.try_multiply(
                        f.col("avg_rank").cast(AVG_RANK_TYPE),
                        f.col(f"missing_n_{corpus}").cast(AVG_RANK_TYPE)
                    ).cast(RANK_SUM_TYPE)
                ).cast(RANK_SUM_TYPE),
                f.try_multiply(
                    f.col("missing_total").cast(SAMPLE_TYPE),
                    f.col(f"n_{corpus}").cast(SAMPLE_TYPE)
                ).cast(RANK_SUM_TYPE)
            ).cast(RANK_SUM_TYPE)
        )))


def calculate_u(final_df: DataFrame, suffix: str):
    # Prepare the dataframe
    final_df = final_df.withColumn(f"n1n2{suffix}",
                                   f.try_multiply(
                                       f.col(f"n_test{suffix}").cast(SAMPLES_SQ_TYPE),
                                       f.col(f"n_baseline{suffix}").cast(SAMPLES_SQ_TYPE)
                                   ).cast(SAMPLES_SQ_TYPE))

    # Calculate the U values
    final_df = _calculate_u_helper(final_df=final_df, corpus="test", suffix=suffix)
    final_df = _calculate_u_helper(final_df=final_df, corpus="baseline", suffix=suffix)

    # Calculate z scores
    final_df = calculate_z(final_df=final_df, suffix=suffix)
    return final_df


def _calculate_u_helper(final_df: DataFrame, corpus: str, suffix: str):
    final_df = (
        final_df
        .withColumn(f"{corpus}_product{suffix}",
                    f.try_multiply(
                        f.col(f"n_{corpus}{suffix}").cast(SAMPLE_TYPE),
                        f.try_add(
                            f.col(f"n_{corpus}{suffix}").cast(SAMPLE_TYPE),
                            f.lit(1).cast(SAMPLE_TYPE)
                        ).cast(SAMPLE_TYPE)
                    ).cast(SAMPLES_SQ_TYPE))
        .withColumn(f"{corpus}_divide{suffix}",
                    f.try_divide(
                        f.col(f"{corpus}_product{suffix}").cast(SAMPLES_SQ_TYPE),
                        f.lit(2).cast(SAMPLES_SQ_TYPE)
                    ).cast(RANK_SUM_TYPE))
        .withColumn(f"{corpus}_subtract{suffix}",
                    f.try_subtract(
                        f.col(f"{corpus}_divide{suffix}").cast(RANK_SUM_TYPE),
                        f.col(f"r_{corpus}{suffix}").cast(RANK_SUM_TYPE)
                    ).cast(RANK_SUM_TYPE))
        .withColumn(f"u_{corpus}{suffix}",
                    f.try_add(
                        f.col(f"n1n2{suffix}").cast(RANK_SUM_TYPE),
                        f.col(f"{corpus}_subtract{suffix}").cast(RANK_SUM_TYPE)
                    ).cast(RANK_SUM_TYPE))
    )
    return final_df


def calculate_z(final_df: DataFrame, suffix: str = "_all"):
    # First, calculate mu_U and sigma_U
    final_df = (final_df
                .withColumn(f"mu_u{suffix}",
                            f.try_divide(
                                f.col(f"n1n2{suffix}").cast(RANK_SUM_TYPE),
                                f.lit(2).cast(RANK_SUM_TYPE)
                            ).cast(RANK_SUM_TYPE))
                .withColumn(f"n1n2_div12{suffix}",
                            f.try_divide(
                                f.col(f"n1n2{suffix}").cast(MAX_DECIMAL_TYPE),
                                f.lit(12).cast(MAX_DECIMAL_TYPE)
                            ).cast(MAX_DECIMAL_TYPE))
                .withColumn(f"n{suffix}",
                            f.try_add(
                                f.col(f"n_test{suffix}").cast(SAMPLE_TYPE),
                                f.col(f"n_baseline{suffix}").cast(SAMPLE_TYPE)
                            ).cast(SAMPLE_TYPE))
                .withColumn(f"tied_rank_correction{suffix}",
                            f.try_divide(
                                f.col(f"tied_rank_sum{suffix}").cast(MAX_DECIMAL_TYPE),
                                f.try_multiply(
                                    f.col(f"n{suffix}").cast(MAX_DECIMAL_TYPE),
                                    f.try_subtract(
                                        f.col(f"n{suffix}").cast(MAX_DECIMAL_TYPE),
                                        f.lit(1).cast(MAX_DECIMAL_TYPE)
                                    ).cast(MAX_DECIMAL_TYPE)
                                ).cast(MAX_DECIMAL_TYPE)
                            ).cast(MAX_DECIMAL_TYPE))
                .withColumn(f"sigma_u{suffix}",
                            f.sqrt(
                                f.try_multiply(
                                    f.col(f"n1n2_div12{suffix}").cast(MAX_DECIMAL_TYPE),
                                    f.try_subtract(
                                        f.try_add(
                                            f.col(f"n{suffix}").cast(MAX_DECIMAL_TYPE),
                                            f.lit(1).cast(MAX_DECIMAL_TYPE)
                                        ).cast(MAX_DECIMAL_TYPE),
                                        f.col(f"tied_rank_correction{suffix}").cast(MAX_DECIMAL_TYPE)),
                                ).cast(MAX_DECIMAL_TYPE)
                            ).cast(MAX_DECIMAL_TYPE))
                )

    # Second, calculate the z-score
    final_df = (
        final_df
        .withColumn(f"u{suffix}",
                    f.least(
                        f.col(f"u_test{suffix}").cast(RANK_SUM_TYPE),
                        f.col(f"u_baseline{suffix}").cast(RANK_SUM_TYPE)
                    ).cast(RANK_SUM_TYPE))
        .withColumn(f"z{suffix}",
                    f.try_divide(
                        f.try_subtract(
                            f.col(f"u{suffix}").cast(MAX_DECIMAL_TYPE),
                            f.col(f"mu_u{suffix}").cast(MAX_DECIMAL_TYPE)
                        ).cast(MAX_DECIMAL_TYPE),
                        f.col(f"sigma_u{suffix}").cast(MAX_DECIMAL_TYPE)
                    ).cast(MAX_DECIMAL_TYPE))
    )
    return final_df


def run(warning: str, data_path: str, job_id: str, tag_chunk_size=50, use_scipy: bool = False):
    # 1. Get the tags to be tested as well as optional excluded ones based on the job_id
    job_json = data_path + f"/job_tags.json"
    if not os.path.exists(job_json):
        print(f"\nNo job tags were found at {job_json}. Please run ao3_tags.aggregate.prepare first.")
        exit(1)

    with open(job_json, "r") as file:
        job_tags = json.load(file)
        test_tags = job_tags[job_id]["test_tags"]
        excluded_tags = job_tags[job_id]["excluded_tags"]  # These include the test tags

    # 3. Load the Parquet files created by ao3_tags/extract/tokenized_chapters
    spark = SparkSession.builder.appName(f'ao3_{warning}_category').getOrCreate()
    chapter_df = spark.read.parquet(f"{data_path}/extract/tokenized_chapters/{warning}/chapters*.parquet")
    word_df = spark.read.parquet(f"{data_path}/extract/tokenized_chapters/{warning}/words*.parquet")

    # 3.1 Add the work_id to each chapter
    get_work_id = f.udf(lambda chapter_id: chapter_id.split("-")[0])
    chapter_df = chapter_df.withColumn("work_id", get_work_id('chapter_id'))

    # 4. Process the chapters with test tags
    # 4.1 Create subsets of tags as they are more manageable; join the results afterwards
    # - union for test_subsets as we want all matches;
    #   A chapter that is in any of the dfs_a has at least one matching tag
    test_subsets = [test_tags[i:i + tag_chunk_size] for i in range(0, len(test_tags), tag_chunk_size)]
    dfs_a = [chapter_df.where(reduce(lambda a, b: a | b, (f.array_contains(f.col('tags'), tag) for tag in subset)))
             for subset in test_subsets]
    chapter_df_a = reduce(lambda a, b: a.union(b), dfs_a)
    chapter_df_a = chapter_df_a.dropDuplicates().withColumn("corpus", f.lit("test")).cache()

    # 5. Process the chapters with tags not in excluded_tags (used as baseline)
    # 5.1 Create subsets of tags as they are more manageable; join the results afterwards
    # - intersect for excluded_subsets as we want no match for any subset;
    #   Only a chapter that is in all dfs_b has no excluded tag
    excluded_subsets = [excluded_tags[i:i + tag_chunk_size] for i in range(0, len(test_tags), tag_chunk_size)]
    dfs_b = [chapter_df.where(~reduce(lambda a, b: a | b, (f.array_contains(f.col('tags'), tag) for tag in subset)))
             for subset in excluded_subsets]
    chapter_df_b = reduce(lambda a, b: a.intersect(b), dfs_b)
    chapter_df_b = chapter_df_b.dropDuplicates().withColumn("corpus", f.lit("baseline")).cache()

    # 6. Put the two separate chapter_df's back together and join them with the word_df;
    # - Calculate ranks per word-pos_tag-combination
    # - The ranks for rows with the same value need to be the average for all of them
    w_rank = Window.partitionBy(["word", "pos_tag"]).orderBy("norm_freq")
    w_avg_rank = Window.partitionBy(["word", "pos_tag", "pyspark_rank"])

    chapter_df = chapter_df_a.union(chapter_df_b)
    corpus_df = (
        word_df.join(chapter_df, how='inner', on='chapter_id')
        .withColumn("norm_freq", f.col("tf") / f.col("chapter_len"))
        .withColumn("row_number", f.row_number().over(w_rank))
        .withColumn("pyspark_rank", f.rank().over(w_rank))
        .withColumn("rank", f.mean(f.col("row_number")).over(w_avg_rank))
    )

    # 7. Calculate the correction for tied ranks (used to adjust the standard deviation)
    tied_ranks_df = (
        corpus_df.groupby(["word", "pos_tag", "rank"]).count()
        .withColumn("tied_rank_diff",
                    f.try_subtract(
                        f.pow(
                            f.col("count").cast(MAX_DECIMAL_TYPE),
                            f.lit(3).cast(MAX_DECIMAL_TYPE)
                        ).cast(MAX_DECIMAL_TYPE),
                        f.col("count").cast(MAX_DECIMAL_TYPE)
                    ).cast(MAX_DECIMAL_TYPE)
                    )
        .groupby(["word", "pos_tag"])
        .agg(f.sum(
            f.col("tied_rank_diff").cast(MAX_DECIMAL_TYPE)
        ).cast(MAX_DECIMAL_TYPE).alias("tied_rank_sum"))
    )

    # 8. Group by word and pos_tag; Aggregate frequencies and ranks in both corpora into arrays
    grouped_df = (
        corpus_df.select("word", "pos_tag", "corpus", "norm_freq", "rank").groupby(["word", "pos_tag"])
        .agg(
            f.collect_list(f.when(f.col("corpus") == "test", f.col("norm_freq"))).alias("freq_test"),
            f.collect_list(f.when(f.col("corpus") == "baseline", f.col("norm_freq"))).alias("freq_baseline"),
            f.collect_list(f.when(f.col("corpus") == "test", f.col("rank"))).alias("rank_test"),
            f.collect_list(f.when(f.col("corpus") == "baseline", f.col("rank"))).alias("rank_baseline"),
        )
        .join(tied_ranks_df, on=["word", "pos_tag"])
    )

    # 9. Calculate rank sums (r_*) and count number of chapters (n_*) with each word-pos_tag-combination
    n_chapters_a = chapter_df_a.count()
    n_chapters_b = chapter_df_b.count()
    rank_sum_df = (
        grouped_df
        .withColumn("r_test", f.aggregate("rank_test", f.lit(0.0), lambda a, b: a + b))
        .withColumn("r_baseline", f.aggregate("rank_baseline", f.lit(0.0), lambda a, b: a + b))
        .withColumn("n_test", f.size(f.col("freq_test")))
        .withColumn("n_baseline", f.size(f.col("freq_baseline")))
        .withColumn("n_test_all", f.lit(n_chapters_a))
        .withColumn("n_baseline_all", f.lit(n_chapters_b))
        .withColumn("missing_n_test", f.try_subtract("n_test_all", "n_test"))
        .withColumn("missing_n_baseline", f.try_subtract("n_baseline_all", "n_baseline"))
    )

    # Keep only words that occur at least once in the test corpus
    rank_sum_df = rank_sum_df.filter(f.col("missing_n_test") < f.col("n_test_all"))

    # 10. Correct for chapters that don't have a given word-pos_tag-combination
    # - Increase the rank sums to account for the missing chapters
    final_df = increase_rank_sum(rank_sum_df=rank_sum_df)
    output_cols = [
        "word", "pos_tag", "z", "z_all",
        "u", "u_test", "u_baseline", "r_test", "r_baseline", "n_test", "n_baseline",
        "u_all", "u_test_all", "u_baseline_all", "r_test_all", "r_baseline_all", "n_test_all", "n_baseline_all",
        "mu_u", "mu_u_all", "sigma_u", "sigma_u_all", "n1n2", "n1n2_all", "n1n2_div12", "n1n2_div12_all", "n", "n_all",
        "tied_rank_correction", "tied_rank_correction_all", "tied_rank_sum", "tied_rank_sum_all"
    ]

    # Use scipy to confirm results; Add 0.0 frequencies for each missing chapter if the test should be performed
    if use_scipy:
        final_df = (
            final_df
            .withColumn("freq_test_all", add_zero_freq_udf(f.col("freq_test"),
                                                           f.col("missing_n_test")))
            .withColumn("freq_baseline_all", add_zero_freq_udf(f.col("freq_baseline"),
                                                               f.col("missing_n_baseline")))
            .withColumn("mwu", mann_whitney_u_udf(f.col("freq_test_all"), f.col("freq_baseline_all")))
        )
        output_cols.append("mwu.*")

    # Calculate the values for U1 and U2 (test and baseline) based on the rank sums
    test_df = calculate_u(final_df=final_df, suffix="")
    test_df = calculate_u(final_df=test_df, suffix="_all")

    # 11. Write the result
    (test_df
     .select(output_cols).repartition(numPartitions=40)
     .write.parquet(f"{data_path}/mannwhitneyu/z_scores/{warning}/{job_id}.parquet", mode="overwrite")
     )


if __name__ == '__main__':
    # Parse the input arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('warning', metavar='w', type=str,
                        help="Warning for which to test word frequencies")
    parser.add_argument('data_path', metavar='d', type=str,
                        help="Path that contains the project's data")
    parser.add_argument('job_id', metavar='i', type=str,
                        help="ID of the job to be run. Used to select tags")
    parser.add_argument('--use_scipy', metavar='s', type=lambda x: bool(strtobool(x)), default=False,
                        required=False,
                        help="Whether to use the Mann-Whitney-U implementation by scipy. "
                             "Not advised for larger collections as objects larger than 2G can't be serialized.")
    args = parser.parse_args()

    # Run the job as specified
    run(warning=args.warning, data_path=args.data_path, job_id=args.job_id, use_scipy=args.use_scipy)
