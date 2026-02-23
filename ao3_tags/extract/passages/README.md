# Extract Passages for Annotations
The scripts in this package collect passages for a specific warning and job ID.
The process can be run using the [run_jobs.sh](run_jobs.sh) script:
```
run_jobs.sh [WARNING] [JOB_ID]
```
Example
```
run_jobs.sh abuse a0123456789
```

## Prerequisites
The jobs require all vocabulary consistency jobs to be completed for the Job ID (Refer to the main [README](../../../README.md)).
In detail, these are:
- Results from the [Mann-Whitney U-test](../../mannwhitneyu)
- Results from the [Jensen Shannon Divergence](../../jensen_shannon_div) calculation  
- Tags for the warning in [tags](../../../resources/tags) or entries in [warnings_fine_open.csv](../../../resources/query_construction/warnings_fine_open.csv).
The tags can be created by running the jobs in the [tags](../../tags)-package.

## Output
- The [sample_words](sample_words.py) script creates files with words that either have a high z-score and/or log ratio in [data](../../../output/data) in the directory `extract/passages/sampled_words/warning`
- The [identify_chapters](identify_chapters.py) script uses the words sampled in the first step to identify chapters that contain them. The chapters are stored in [data](../../../output/data) in the directory `extract/passages/chapter_matches/warning`
- The [retrieve_chapters](retrieve_chapters.py) script takes the identified chapters from the previous step to retrieve their content. The chapter content is stored in[data](../../../output/data) in the directory `extract/passages/chapter_content/warning`
- The [identify_passages](identify_passages.py) script is the final step that collects the passages in each chapter that contain one of the sampled words. The passages are stored in [data](../../../output/data) in the directory `extract/passages/passages/warning/`

The final files are the parquet files from the postprocessing job. They are named after the job ID and contain the following columns:
- `id`:            ID of the passage
- `category_tags`: Tags of the chapter that belong to the warning category
- `matches`:       Array of sampled words contained in the passage
- `passage`:       Passage string
