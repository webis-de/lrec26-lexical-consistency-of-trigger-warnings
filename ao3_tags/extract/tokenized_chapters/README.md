# Extract Tokenized Chapters
The scripts in this package retrieve chapters for a specified warning and tokenize them.
The process can be run using the [run_jobs.sh](run_jobs.sh) script:
```
run_jobs.sh [WARNING]
```
Example
```
run_jobs.sh abuse
```

## Prerequisites
The jobs require tags for the specified warnings in [tags](../../../resources/tags) 
or entries in [warnings_fine_open.csv](../../../resources/query_construction/warnings_fine_open.csv).
The tags can be created by running the jobs in the [tags](../../tags)-package.

## Output
- The [multiprocess.py](multiprocess.py) script creates JSON-L files in [data](../../../output/data) in the directory `extract/tokenized_chapters/warning/jsonl`
- The postprocessing job aggregates the JSON-L files into parquet files in [data](../../../output/data) in the directory `extract/tokenized_chapters/warning/`

**words.parquet**:
- `word`: Lemmatized word
- `pos_tag`: POS_TAG for the word
- `tf`: Raw count of the word in the chapter with `chapter_id`

**chapters.parquet**:
- `chapter_id`: ID of the chapter
- `tags`: List of tags assigned to the chapter
- `chapter_len`: Total number of words in the chapter