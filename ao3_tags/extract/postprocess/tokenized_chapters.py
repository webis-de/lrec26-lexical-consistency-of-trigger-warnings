import json
from pyspark.sql import Row, SparkSession
import pyspark.sql.functions as f


def read_jsonl(spark, file_prefix="words_extracted", suffix=None):
    suffix = suffix or "-*"

    # Load the JSONL file as a text, parse the JSON in each row and turn it into a new DataFrame
    df = spark.read.text(f"{file_prefix}{suffix}.jsonl")
    json_data = df.rdd.map(lambda x: Row(**json.loads(x.value)))
    return spark.createDataFrame(json_data)


# Main function
def run(warning: str, data_path: str, suffix: str = None):
    file_prefix = f"{data_path}/extract/tokenized_chapters/{warning}/jsonl"
    output_prefix = f"{data_path}/extract/tokenized_chapters/{warning}"

    spark = SparkSession.builder.appName(f'ao3_{warning}_aggregation').getOrCreate()

    # 1. Load the JSONL files created by multiprocessing
    chapter_df = read_jsonl(spark=spark, file_prefix=f"{file_prefix}/chapters_extracted", suffix=suffix)
    word_df = read_jsonl(spark=spark, file_prefix=f"{file_prefix}/words_extracted", suffix=suffix)

    # 2. Group the word_df by chapter_id, word and pos_tag and sum over the word frequency column
    # Ensure that the frequencies are deduplicated in case the multiprocessing ran over multiple iterations
    word_df = (word_df
               .groupBy(["chapter_id", "word", "pos_tag"])
               .agg(f.sum("tf").alias("tf"))
               .join(chapter_df.groupBy("chapter_id").count(), how="left", on="chapter_id")
               .withColumn("tf", f.try_divide(f.col("tf"), f.col("count")))
               )
    word_df = word_df.select("chapter_id", "word", "pos_tag", "tf").cache()

    # 3. Sum the word frequency per chapter_id to get the chapter length
    chapter_len = word_df.groupBy(f.col("chapter_id")).agg(f.sum('tf').alias('chapter_len'))
    chapter_df = (chapter_df
                  .groupBy(f.col("chapter_id")).agg(f.first("tags").alias("tags"))
                  .join(chapter_len, how="left", on="chapter_id"))

    # 4. Write the results
    suffix = f"-{suffix}" if suffix else ""
    word_df.write.parquet(f"{output_prefix}/words{suffix}.parquet", mode="overwrite")
    chapter_df.write.parquet(f"{output_prefix}/chapters{suffix}.parquet", mode="overwrite")
    print('\n'*2 + f'Saved results under {output_prefix} with suffixes "_words/chapters.parquet"' + '\n'*2)


if __name__ == '__main__':
    # Parse the input arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('warning', metavar='w', type=str,
                        help="Warning for which to aggregate extraction results")
    parser.add_argument('data_path', metavar='d', type=str,
                        help="Path that contains the project's data")
    args = parser.parse_args()

    # Run the job as specified
    run(warning=args.warning, data_path=args.data_path)

