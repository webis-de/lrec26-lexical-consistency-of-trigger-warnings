#!/bin/bash
warning=$1
job_type=$2
job_id=$3

# Set the package directory and get the data_path
package_dir="$(dirname $(dirname $(readlink -f $0)))/ao3_tags"
. ${package_dir}/utils/utils.sh --source-only
eval $( parse_yaml ${package_dir}/conf/config.yaml )

# Get the data_path from ao3_tags
PATHS=$(python3 ${package_dir}/__init__.py)
IFS=$"|" read -r resource_path data_path<<< ${PATHS}

# 1. Check if the aggregated files exist
extract_directory=${data_path}/extract/tokenized_chapters/${warning}/
word_parquets=$(find ${extract_directory} -type d -regextype posix-extended -regex '.*/words.*.parquet' | wc -l)
chapter_parquets=$(find ${extract_directory} -type d -regextype posix-extended -regex '.*/chapters.*.parquet' | wc -l)

if (($word_parquets > 0))
then
  if (($word_parquets != $chapter_parquets))
  then
    echo
    echo "The number of word parquet files in ${extract_directory} is different from the number of chapter parquet files."
    echo "Please verify the files."
    echo
    exit 1
  fi
else
  echo
  echo "No word parquet files were found in ${extract_directory}. Please run the extract.tokenized_chapters steps first."
  exit 1
fi

# 2. Run prepare_jobs.py if no job_id was given
if [ -z "${job_id}" ]
then
  python -m ao3_tags.prepare_jobs ${warning}
  echo
  read -p 'Please enter the job id: ' job_id
fi

# 3. Run the run_job.sh for the specified job_type and id
${package_dir}/${job_type}/run_jobs.sh ${warning} ${job_id}