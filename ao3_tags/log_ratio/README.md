# Calculate Log Ratios
The scripts in this package calculate the log ratios of corpus-level term frequencies between the category corpus and baseline corpus.
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
- The term frequencies need to be calculated by [term_frequencies.py](../jensen_shannon_div/term_frequencies.py)

## Output
The script creates files named after the `job_id` in subdirectory of [data/log_ratio](../../../output/data).
It is run [spark_job.sh](spark_job.sh).

### [log_ratios.py](log_ratios.py)
This job builds on the term frequencies to calculate log ratios for each word. The files are created in the `log_ratios` subdirectory.
The columns are:

- `word`: Lemmatized word
- `pos_tag`: POS_TAG for the word
- `norm_freq`: Normalized term frequency in the test (category) corpus
- `n_words`, `n_chapters`, `n_works`: Analogous to above; Numbers for the test (category) corpus
- `log_ratio`: Binary log of the ratio of `norm_freq` and `norm_freq_baseline`