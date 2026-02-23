# Calculate the Jensen Shannon Divergence
The scripts in this package calculates the Jensen Shannon Divergence between the category corpus and baseline corpus.
For each term, it records the contribution to that divergence as defined by [Pechenick et al.](https://arxiv.org/pdf/1501.00960)
The process can be run using the [run_jobs.sh](run_jobs.sh) script:
```
run_jobs.sh [WARNING] [JOB_ID]
```
Example
```
run_jobs.sh abuse a0123456789
```

## Prerequisites
- The jobs require tokenized chapters for the specified warning in [data](../../../../output/data) in the directory `extract/warning/`
  - The files are created by the [extract.tokenized_chapters](../extract/tokenized_chapters) package
- The job ID needs to be created with the [prepare_jobs.py](../prepare_jobs.py) script
  - The [prepare_jobs.py](../prepare_jobs.py) script needs manual labels for the `warning.csv` file in [tags](../../resources/tags) 

## Output
The two scripts create files named after the `job_id` in subdirectories of [data/jensen_shannon_div](../../../output/jensen_shannon_div).
Both scripts are run using [spark_job.sh](spark_job.sh).

### [term_frequencies.py](term_frequencies.py)
The files are created in the `term_frequencies` subdirectory. For each `job_id`, two files are created:
- `[job_id]_words_test.parquet`: Term frequencies in the test (category) corpus
- `[job_id]_words_baseline.parquet`: Term frequencies in the baseline corpus

Both files have the same columns:

- `word`: Lemmatized word
- `pos_tag`: POS_TAG for the word
- `tf`: Raw count in the entire corpus
- `num_words`: Number of words in the corpus (Same for all rows)
- `num_chapters`: Number of chapters in the corpus (Same for all rows)
- `num_works`: Number of works in the corpus (Same for all rows; A work can contain multiple chapters)


### [jsd.py](jsd.py)
This job builds on the term frequencies to calculate contribution to Jensen Shannon divergence for each word. 
The files are created in the `jsd` subdirectory.
The columns are:

- `word`: Lemmatized word
- `pos_tag`: POS_TAG for the word
- `p`: Probability of the term in the test (category) corpus (`tf` / `num_words`)
- `n_words`, `n_chapters`, `n_works`: Analogous to above; Numbers for the test (category) corpus
- `jsd`: Contribution to the Jensen Shannon Divergence based on `p` and `p_baseline`
