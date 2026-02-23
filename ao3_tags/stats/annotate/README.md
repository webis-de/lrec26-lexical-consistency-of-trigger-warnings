# Statistical Tests on Annotations

## Prerequisites
The jobs in the package [annotate](../../annotate) are completed for the Job ID

## Output
The output will be created in [stats/annotations/warning](../../../output/stats).

### [distributions.py](distributions.py)
Create parquet files that contain the distribution of positive annotations per word, word group, and quantiles of z-score and log ratio.
```
python -m ao3_tags.stats.annotate.distributions
```
When prompted, provide the warning (e.g., `abuse`) and job id for which to create distributions of annotations.

### [mannwhitneyu.py](mannwhitneyu.py)
Apply a Mann-Whitney U-test to the distribution of annotations per word by comparing it against the distribution of annotations on random passages.
```
python -m ao3_tags.stats.annotate.mannwhitneyu
```
When prompted, provide the warning (e.g., `abuse`) and job id for which to apply the Mann-Whitney U-test.

### [mannwhitneyu_corr.py](mannwhitneyu_corr.py)
Calculate the correlation between z-score/log ratio and the results of the Mann-Whitney U-test. Applied to all categories for a warning
```
python -m ao3_tags.stats.annotate.mannwhitneyu_corr
```
