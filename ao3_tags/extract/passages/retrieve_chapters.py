import click
import math
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from tqdm import tqdm

from ao3_tags import DATA_PATH
from ao3_tags.utils.queries import CLIENT

index = 'ao3-v3-chapters'


def main(warning, job_id, batch_size: int = 512):
    # Load the dataframe of chapter IDs that have at least one category tag
    chapter_dir = DATA_PATH / "extract" / "passages" / "chapter_matches" / warning / job_id
    dataset = pq.ParquetDataset(
        chapter_dir,
        filters=~ds.field("category_tags").is_null()
    )
    df = dataset.read().to_pandas()

    # Ensure that the output directory exists
    output_dir = DATA_PATH / "extract" / "passages" / "chapter_content" / warning
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / f"{job_id}.parquet"

    # Iterate over the IDs in steps of batch_size and write the chapter content to a parquet file
    i = 0
    ids = df.iloc[i * batch_size:(i + 1) * batch_size].index.to_list()
    table = pa.Table.from_pandas(ids_to_df(ids))
    with pq.ParquetWriter(output_file, table.schema) as writer:
        writer.write_table(table)

        for i in tqdm(range(1, math.ceil(df.shape[0] / batch_size)), desc=f"Retrieving Chapters for {job_id}"):
            ids = df.iloc[i * batch_size:(i + 1) * batch_size].index.to_list()
            table = pa.Table.from_pandas(ids_to_df(ids))
            writer.write_table(table)
    print(f"Chapter content has been written to {output_file}.")


def ids_to_df(ids) -> pd.DataFrame:
    response = CLIENT.mget(index=index, body={'ids': ids})
    return pd.DataFrame([{"chapter_id": h["_id"], "chap_content": h["_source"]["chap_content"]}
                         for h in response.body.items().mapping["docs"]])


if __name__ == "__main__":
    import argparse

    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("warning", metavar="w", type=str, default="abuse",
                        help='Warning for which to retrieve chapters identified in the previous step.')
    parser.add_argument("job_id", metavar="i", type=str, default="29d13398b5",
                        help='ID of the job used to tokenize chapters. Sets warning category')
    args = parser.parse_args()
    main(warning=args.warning, job_id=args.job_id)