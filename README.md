# Lexical Consistency of Trigger Warnings
This package provides functionalities to study the consistency of trigger warnings applied by authors.
It is based on the [Webis Trigger Warning Corpus 2022](https://zenodo.org/records/7976807).

## Getting Started
### Cloning the repository
The repository should be cloned to a cluster directory that can be accessed by a Spark cluster and GPU nodes.
Our implementation assumes a Kubernetes-based Spark cluster and Slurm for job scheduling.

### Setting configurations in the YAML-file
After cloning the repository, update the configurations in [config.yaml](ao3_tags/conf/config.yaml) to reflect your deployment:
1. **Path parameters**:
   1. The `project_dir` sets the highest level of the project. Ensure that last directory is equal to this repository: `LOCATION/ON/YOUR/CLUSTER/acl25-lexical-consistency-of-trigger-warnings`
   2. The `tag_path` is the location of JSON-L files with the tag graph constructed by Wiegmann et al.: https://github.com/MattiWe/acl23-trigger-warning-assignment/

2. **API-Keys and Elastic instance**
   1. `kibana.key`: Key for an Elasticsearch cluster that contains the Webis Trigger Warning Corpus 2022
   2. `kibana.host`: Host of your Elastic instance
   3. `huggigingface.key`: Key for the Huggingface inference API. Optional: Only needed for the [annotation job](./ao3_tags/annotate/hugging_face_api.py) that uses Hugging Face

### Install the package
After performing the configurations, install the Python package using pip
```
pip install .
```

## Collecting tags
| Job # | Package                                                             | Description                                                                                                                                                            |
|-------|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | [tags](./ao3_tags/tags)                                             | Collect all tags for a warning. Either by parsing the graph of tags in the tag_path of [config.yaml](./ao3_tags/conf/config.yaml) or directly from an AO3 tag website. |

The tags package creates a CSV-file of tags for the specified root node. This needs to be manually annotated following the tag categorization for your use case. 
The process is described in the [README](ao3_tags/tags/README.md) of the tags-package.

## Tokenizing chapters
| Job # | Package                                                             | Description                                                                        |
|-------|---------------------------------------------------------------------|------------------------------------------------------------------------------------|
| 2     | [extract.tokenized_chapters](./ao3_tags/extract/tokenized_chapters) | Tokenize chapters that are tagged for a specified warning                          |

After the tokenizing the chapters, Parquet-files called `words.parquet` and `chapters.parquet` exist in the [data](./output/data)-directory in `data/extract/tokenized_chapters`.
These can be used to perform the consistency tests.

## Vocabulary Consistency
A large part of the package is used to test the consistency between chapter tags and the vocabulary of these chapters.
The focus lies on tags that indicate potentially triggering content.

The packages require a **warning** and a set of **categories** of that warning. Example:

- **warning**: Abuse
- **Categories**: Emotional abuse, physical abuse, and sexual abuse 

Using a predefined category vocabulary (e.g. words associated with emotional abuse), the scripts test if chapters 
that are tagged for that category use words from their vocabulary significantly more often than other chapters.

### Pipeline script
If you already created a [vocabulary](./ao3_tags/vocabulary) for the category that you want to perform the consistency tests on, you can run the following script. 
Otherwise, follow the detailed steps below.
```
./scripts/vocabulary_pipeline [WARNING]
```
This script performs the detailed steps below one after the other
1. Create a Job ID
2. Perform Mann-Whitney U-test on document-level term frequencies
3. Calculate Jensen Shannon Divergence for all terms
4. Calculate log ratio for all terms
5. Combine results of Mann-Whitney U-test, Jensen Shannon Divergence, and log ratio into a single `.csv`-file

The detailed steps are as follows

### Detailed steps
<details>
   <summary><b>Show details</b></summary>

### Job ID
The first step before performing tags is creating a **job ID**. This is done by running the [prepare_jobs.py](./ao3_tags/prepare_jobs.py) with a `warning`:

Example
```
python -m ao3_tags.prepare_jobs abuse
```
This opens a dialogue to select the categories for which to construct a corpus. The options depend on the columns in the `[warning].csv` in [tags](./resources/tags).
Refer to the [README](ao3_tags/tags/README.md) of the tags-package.

After completing the dialogue, a job ID is created that can be used in all downstream application to reference this configuration.
All job IDs are stored in [data](../output/data) in the `job_metadata.csv`-file.

### Term Frequency Measures
| Job # | Package                                              | Description                                                                        |
|-------|------------------------------------------------------|------------------------------------------------------------------------------------|
| 3     | [mannwhitneyu](./ao3_tags/mannwhitneyu)              | Perform significance testing on the vocabulary terms using the Mann-Whitney U-test |     
| 4     | [jensen_shannon_div](./ao3_tags/jensen_shannon_div)  | Calculate the Jensen Shannon Divergence for the vocabulary terms                   |
| 5     | [log_ratio](./ao3_tags/log_ratio)                    | Calculate the log ratio for the vocabulary terms                                   |


### Significance Testing
| Job # | Package                                         | Description                                                         |
|-------|-------------------------------------------------|---------------------------------------------------------------------|
| 6     | [vocabulary](./ao3_tags/vocabulary)             | Collect vocabulary terms for the statistical tests                  |
| 7     | [stats.vocabulary](./ao3_tags/stats/vocabulary) | Perform statistical tests on the consistency of tags and vocabulary |

</details>

## Annotation Experiments
| Job # | Package                                         | Description                                                                        |
|-------|-------------------------------------------------|------------------------------------------------------------------------------------|
| 1     | [extract.passages](./ao3_tags/extract/passages) | Retrieve passages for a category that contain warning-related terms                |
| 2     | [annotate](./ao3_tags/annotate)                 | Sample passages for annotation and get annotations with sociodemographic prompting |
| 3     | [stats.annotate](./ao3_tags/stats/annotate)     | Perform statistical tests on the annotations                                       |


# Additional directories
- [models](./models): Directory to store model weights of Huggingface models
- [output](./output): Directory to save output of any jobs run by the package
