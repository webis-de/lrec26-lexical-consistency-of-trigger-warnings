# Statistical Tests on Vocabulary

## Prerequisites
The jobs in the following packages are completed for the job ID:
- [mannwhitneyu](../../mannwhitneyu)
- [jensen_shannon_div](../../jensen_shannon_div)
- [log_ratio](../../log_ratio)

## Output
The output will be created in [stats/vocabulary/warning](../../../../output/stats).

### [join_results.py](join_results.py)
```
python -m ao3_tags.stats.vocabulary.join_results [WARNING] [JOB_ID]
```
Example
```
python -m ao3_tags.stats.vocabulary.join_results abuse a0123456789
```
This creates a file named after the job id that combines Jensen Shannon Divergence (jsd) and z-scores from the Mann-Whitney U-test in one `.csv`-file.
The terms in the files are those from the category vocabulary that belongs to the job id.

### [ttest_cohen_d.py](ttest_cohen_d.py)
```
python -m ao3_tags.stats.vocabulary.ttest_cohen_d [WARNING]
```
Example
```
python -m ao3_tags.stats.vocabulary.ttest_cohen_d abuse
```
- Perform a one sample t-test on the mean of the z-scores being significantly different from 0.
- The script uses the output from [join_results](join_results.py)