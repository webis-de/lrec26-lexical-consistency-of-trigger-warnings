#!/bin/bash
warning=$1
job_id=$2

# 1. Set the script directory
script_dir=$(dirname $(readlink -f $0))

# 2. Run prepare_jobs.py if no job_id was given
if [ -z "${job_id}" ]
then
  python -m ao3_tags.prepare_jobs ${warning}
  echo
  read -p 'Please enter the job id: ' job_id
fi

# 3. Run the packages in order
${script_dir}/run_jobs.sh ${warning} "mannwhitneyu" ${job_id}
${script_dir}/run_jobs.sh ${warning} "jensen_shannon_div" ${job_id}
${script_dir}/run_jobs.sh ${warning} "log_ratio" ${job_id}
python3 -m ao3_tags.stats.vocabulary.join_results ${warning} ${job_id}
python3 -m ao3_tags.stats.vocabulary.ttest_cohen_d ${warning}
