from functools import reduce
import json
import os
from pandas import read_csv
from pyspark.sql import SparkSession
import pyspark.sql.functions as f


def process_chapter_df(chapter_df):
    num_words = chapter_df.agg(f.sum('chapter_len')).first()[0]
    num_chapters = chapter_df.count()
    num_works = chapter_df.select(f.countDistinct("work_id")).first()[0]
    return (chapter_df
            .withColumn('num_words', f.lit(num_words))
            .withColumn('num_chapters', f.lit(num_chapters))
            .withColumn('num_works', f.lit(num_works))
                    )


def write_word_df(chapter_df, word_df, out_prefix, file_suffix):
    # Get all word frequencies from chapters that are part of the provided chapter_df
    word_df = word_df.join(chapter_df, how='inner', on='chapter_id')

    # Aggregate by pos_tag and word, sum tf and select the first value for other agg columns; write the result
    final_df = (word_df
                .groupBy(['pos_tag', 'word'])
                .agg(f.sum('tf').alias('tf'),
                     f.first('num_words').alias('num_words'),
                     f.first('num_chapters').alias('num_chapters'),
                     f.first('num_works').alias('num_works')
                     )
                )
    final_df.write.parquet(f"{out_prefix}_{file_suffix}.parquet", mode='overwrite')


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


# Main function
def run(warning: str, data_path: str, job_id: str, tag_chunk_size=50):
    # 1. Get the tags to be tested as well as optional excluded ones based on the job_id
    job_json = data_path + f"/job_tags.json"
    if not os.path.exists(job_json):
        print(f"\nNo job tags were found at {job_json}. Please run ao3_tags.prepare_jobs.py first.")
        exit(1)

    with open(job_json, "r") as file:
        job_tags = json.load(file)
        test_tags = job_tags[job_id]["test_tags"]
        excluded_tags = job_tags[job_id]["excluded_tags"]   # These include the test tags

    # 3. Load the Parquet files created by ao3_tags/extract/chapters.py
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
    chapter_df_a = chapter_df_a.dropDuplicates()

    # 4.2 Sum the chapter_len column to get the total number of words for each group; Also store number of chapter/works
    chapter_df_a = process_chapter_df(chapter_df=chapter_df_a)

    # 4.3 Join the chapter_df with the word_df to get word frequencies for chapters with test_tags; Write the result
    out_prefix = f"{data_path}/jensen_shannon_div/term_frequencies/{warning}/{job_id}_words"
    write_word_df(chapter_df=chapter_df_a, word_df=word_df, out_prefix=out_prefix, file_suffix="test")

    # 5. Process the chapters with tags not in excluded_tags (used as baseline)
    # 5.1 Check if a previous job has the same excluded_tags to not redo the baseline with the same result
    baseline_id = get_baseline_id(data_path=data_path, job_id=job_id)
    baseline_path = f"{data_path}/jensen_shannon_div/term_frequencies/{warning}/{baseline_id}_words_baseline.parquet"
    if not os.path.exists(baseline_path):
        # 5.2 Create subsets of tags as they are more manageable; join the results afterwards
        # - intersect for excluded_subsets as we want no match for any subset;
        #   Only a chapter that is in all dfs_b has no excluded tag
        excluded_subsets = [excluded_tags[i:i + tag_chunk_size] for i in range(0, len(test_tags), tag_chunk_size)]
        dfs_b = [chapter_df.where(~reduce(lambda a, b: a | b, (f.array_contains(f.col('tags'), tag) for tag in subset)))
                 for subset in excluded_subsets]
        chapter_df_b = reduce(lambda a, b: a.intersect(b), dfs_b)
        chapter_df_b = chapter_df_b.dropDuplicates()

        # 5.3 Sum the chapter_len column to get the total number of words for each group; Store num of chapter/works
        chapter_df_b = process_chapter_df(chapter_df=chapter_df_b)

        # 5.4 Join the chapter_df with the word_df to get word frequencies for chapters with test_tags; Write the result
        write_word_df(chapter_df=chapter_df_b, word_df=word_df, out_prefix=out_prefix, file_suffix="baseline")

    print('\n'*2 + f'Saved results under {out_prefix} with suffixes "_test/baseline.parquet"' + '\n'*2)


if __name__ == '__main__':
    # Parse the input arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('warning', metavar='w', type=str,
                        help="Warning for which to aggregate word frequencies")
    parser.add_argument('data_path', metavar='d', type=str,
                        help="Path that contains the project's data")
    parser.add_argument('job_id', metavar='i', type=str,
                        help="ID of the job to be run. Used to select tags")
    args = parser.parse_args()

    # Run the job as specified
    run(warning=args.warning, data_path=args.data_path, job_id=args.job_id)

