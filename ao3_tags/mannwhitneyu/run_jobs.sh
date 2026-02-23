#!/bin/bash
# Meant to be called by ao3_tags/run_jobs.sh
warning=$1
job_id=$2

dir="$(dirname $(readlink -f $0))"

kubectl auth can-i create pods
${dir}/spark_job.sh ${warning} ${job_id}
python -m ao3_tags.mannwhitneyu.effect_sizes ${warning} ${job_id}
python -m ao3_tags.mannwhitneyu.significant_terms ${warning} ${job_id}