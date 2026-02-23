# Apply Mann-Whitney U-test
The scripts in this package perform the Mann-Whitney U-test on document-level term frequencies to determine which terms occur significantly more in the category corpus.
The process can be run using the [run_jobs.sh](run_jobs.sh) script:
```
run_jobs.sh [WARNING] [JOB_ID]
```
Example
```
run_jobs.sh abuse a0123456789
```

## Prerequisites
- The jobs require tokenized chapters for the specified warning in [data](../../output/data) in the directory `extract/warning/`
  - The files are created by the [extract.tokenized_chapters](../extract/tokenized_chapters) package
- The job ID needs to be created with the [prepare_jobs.py](../prepare_jobs.py) script
  - The [prepare_jobs.py](../prepare_jobs.py) script needs manual labels for the `warning.csv` file in [tags](../../resources/tags) 

## Output
The three different scripts build on each other and all create files named after the `job_id` in subdirectories of [data/mannwhitneyu](../../output/data):

### [z_scores.py](z_scores.py)
This script is run using [spark_job.sh](spark_job.sh). The files are created in the `z_scores` subdirectory. 
The files contain a lot of columns, many of which are helper columns to verify the calculation process.

The suffix `_all` indicates that all chapters (also those in which the word does not occur) were considered. 
Those are the values to work with in downstream tasks.


### [effect_sizes.py](effect_sizes.py)
This script takes the `z_scores`-output and applies post-processing. Files are stored in the `effect_sizes` subdirectory.
The numbers are formatted correctly and helper columns are dropped.

- `word`: Lemmatized word
- `pos_tag`: POS_TAG for the word
- `z_all`: z-score for the Mann-Whitney U-test
- `u_all`: U-statistic to the z-score
- `n_test_all`: Number of chapters in the test (category) corpus
- `n_baseline_all`: Number of chapters in the baseline corpus
- `std_effect_size_all`: Standardized effect size for the z-score
- `cl_effect_size_all`: Common language effect size for the U-statistic
- `z_two_sided_all`: z-score that is made positive/negative based on the direction of the `cl_effect_size_all` (> 0.5 or < 0.5) 


### [significant_terms.py](significant_terms.py)
This script calculates p-values for the z-scores and saves only significant terms. Files are stored in the `significant_terms` subdirectory.
The columns are largely identical with the output of `effect_sizes.py`. The only difference is that p-values were added.
