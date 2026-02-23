from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pandas import read_csv


def get_baseline_id(data_path: str, job_id: str):
    """
    Function to get a job id that has the same excluded_tags and thus the same baseline probabilities
    """
    df = read_csv(data_path + "/job_metadata.csv")
    row = df.loc[df["id"] == job_id].iloc[0]
    test_categories = row["test_categories"]
    excluded_categories = row["excluded_categories"]

    # Return the first ID as the jobs are sorted by creation time
    return df.loc[(df["test_categories"] == test_categories) &
                  (df["excluded_categories"] == excluded_categories)]["id"].iloc[0]


def calculate_norm_frequencies(df):
    """
    Aggregate by word and pos_tag and normalized frequencies
    """
    df = (
        df.groupBy(['word', 'pos_tag'])
        .agg(f.sum('tf').alias('tf'),
             f.first('num_words').alias('num_words'),
             f.first('num_chapters').alias('num_chapters'),
             f.first('num_works').alias('num_works')
             )
        .withColumn("norm_freq", f.col("tf") / f.col("num_words"))
    ).select("word", "pos_tag", "norm_freq", "num_words", "num_chapters", "num_works")
    return df


def calculate_log_ratio(df_a, df_b):
    """
    Calculate the log ratio for each word-pos_tag-combination; log_2(norm_freq / norm_freq_baseline)
    """
    # Rename column (consistency with mannwhitneyu)
    renaming_cols = ["num_words", "num_chapters", "num_works"]
    mapping = dict(zip(renaming_cols, [c.replace("num", "n") for c in renaming_cols]))
    df_a = df_a.select([f.col(c).alias(mapping.get(c, c)) for c in df_a.columns])
    df_b = df_b.select([f.col(c).alias(mapping.get(c, c)) for c in df_b.columns])

    # Rename to avoid duplicate names
    shared_cols = ["norm_freq", "n_words", "n_chapters", "n_works"]
    mapping = dict(zip(shared_cols, [c + "_baseline" for c in shared_cols]))
    df_b = df_b.select([f.col(c).alias(mapping.get(c, c)) for c in df_b.columns])

    # Join and calculate the log ratios
    df = df_a.join(df_b, on=['word', 'pos_tag'], how='left')
    df = (df
          .withColumn("log_ratio",
                      f.log(2.0,
                            f.try_divide(
                                f.col("norm_freq"),
                                f.col("norm_freq_baseline")
                            )
                            )
                      )
          )
    return df


# Main function
def run(warning: str, data_path: str, job_id: str):
    # 1. Get the dfs to be tested and the baseline (the baseline can be reused from a previous job)
    spark = SparkSession.builder.appName(f'ao3_{warning}_group_probabilities').getOrCreate()
    input_prefix = f"{data_path}/jensen_shannon_div/term_frequencies/{warning}/"
    df_a = spark.read.parquet(input_prefix + f"{job_id}_words_test.parquet")

    baseline_id = get_baseline_id(data_path=data_path, job_id=job_id)
    df_b = spark.read.parquet(input_prefix + f"{baseline_id}_words_baseline.parquet")

    # 2. Add normalized frequencies for each word (on corpus level)
    df_a = calculate_norm_frequencies(df_a)
    df_b = calculate_norm_frequencies(df_b)

    # 3. Calculate the log ratios and write the results
    df = calculate_log_ratio(df_a, df_b)

    # 4. Write the results
    output_file = f"{data_path}/log_ratio/log_ratios/{warning}/{job_id}.parquet"
    df.repartition(numPartitions=40).write.parquet(output_file, mode="overwrite")
    print('\n' * 2 + f'Saved results under {output_file}"' + '\n' * 2)


if __name__ == '__main__':
    # Parse the input arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('warning', metavar='w', type=str,
                        help="Warning for which to calculate log ratios")
    parser.add_argument('data_path', metavar='d', type=str,
                        help="Path that contains the project's data")
    parser.add_argument('job_id', metavar='i', type=str,
                        help="ID of the job to be run. Used to identify files")
    args = parser.parse_args()

    # Run the job as specified
    run(warning=args.warning, data_path=args.data_path, job_id=args.job_id)

