from elasticsearch import Elasticsearch
from nltk.stem.porter import PorterStemmer
from pandas import read_parquet
from pathlib import Path
from pyspark.sql import SparkSession
import re
from tqdm import tqdm
from typing import Sequence

from ao3_tags import DATA_PATH, RESOURCE_PATH
from ao3_tags.utils.queries import CLIENT

# =============================================================
#           Collect character names
# =============================================================
def gather_names(api_key: str, out_file: str = "ao3_characters.txt"):
    out_path = DATA_PATH / out_file

    results = CLIENT.search(index='ao3-v3-chapters',
                            fields=[{'field': 'character'}, {"field": "fandom"}],
                            source=False,
                            query={"exists": {"field": "character"}},
                            size=10000,
                            scroll="10m"
                            )

    sid = results['_scroll_id']
    scroll_size = results['hits']['total']['value']
    total_size = results['hits']['total']['value']

    with tqdm(total=total_size, desc="Processing chapters") as pbar:
        while scroll_size > 0:
            results = CLIENT.scroll(scroll_id=sid, scroll='10m')
            sid = results['_scroll_id']

            # Get the number of results for the current page
            scroll_size = len(results['hits']['hits'])

            # Process the results
            for hit in results['hits']['hits']:
                characters = hit["fields"]["character"]
                _process_characters(char_list=characters, out_path=out_path)

            pbar.update(scroll_size)

def _process_characters(char_list: Sequence[str], out_path: Path):
    for c in char_list:
        # Remove fandom in brackets; Remove common content like "|", "Character" " - "
        c = re.sub(r'\([^)]*\)', '', c)
        c = c.replace("|", "").replace("Character", "").replace(" - ", "")

        # Write parts with at least 3 characters
        with open(out_path, "a") as file:
            for line in [part for part in c.split(" ") if len(part) > 2]:
                file.write(f"{line.lower()}\n")


# =============================================================
#           Aggregate character names
# =============================================================
filter_list = ['the', 'original', 'female', 'male', 'you', 'girl', 'boy', 'man', 'woman', 'lady', 'mister',
               'sister', 'brother', 'aunt', 'uncle', 'mother', 'father', 'son', 'daughter', 'husband', 'wife',
               'child', 'family', 'parents', 'siblings', 'teacher', 'student', 'doctor', 'nurse', 'engineer', 'artist',
               'police', 'fireman', 'manager', 'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'black',
               'white', 'gray', 'brown', 'violet', 'maroon', 'navy', 'turquoise', 'gold', 'silver', 'reader', 'all',
               'snow', 'and', 'tag', 'tags', 'characters', 'monster', 'one', 'added', 'present', 'for', 'team',
               'master', 'mentioned', 'everyone', 'human', 'various', 'others', 'unit', 'minor', 'background',  'evil',
               'america', 'deceit', 'club',  'dog', 'cat', 'league']

def aggregate_names(in_file: str = "ao3_characters.txt", out_file: str = "ao3_characters.parquet"):
    in_path = str(DATA_PATH / in_file)
    out_path = str(DATA_PATH / out_file)

    # Load the text file
    spark = SparkSession.builder.appName('WordCount').getOrCreate()
    txt_file = spark.read.text(in_path).withColumnRenamed('value', 'name')
    names = txt_file.filter(~txt_file.name.isin(filter_list))

    # Count the occurrences of each name and write the results to a parquet file
    name_counts = names.groupBy('name').count()
    name_counts.write.parquet(out_path)

def extract_common_names(in_file: str = "ao3_characters.parquet", out_file: str = "ao3-characters-automated.txt",
                         k: int = 10_000, add_stemmed_version=True):
    in_path = str(DATA_PATH / in_file)
    out_path = str(RESOURCE_PATH / f"wordlists/{out_file}")
    df = read_parquet(in_path)
    names = df.sort_values("count", ascending=False).iloc[:k]["name"].tolist()
    with open(out_path, "w") as f:
        for name in names:
            f.write(name + "\n")

    if add_stemmed_version:
        ps = PorterStemmer()
        stemmed_names = [ps.stem(name) for name in names]
        with open(out_path.replace(".txt", "-stemmed.txt"), "w") as f:
            for name in stemmed_names:
                f.write(name + "\n")