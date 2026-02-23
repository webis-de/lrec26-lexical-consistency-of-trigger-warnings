#!/bin/bash
# Meant to be called by ao3_tags/run_jobs.sh
warning=$1
job_id=$2

postprocess_dir="$(dirname $(dirname $(readlink -f $0)))/postprocess"

python -m ao3_tags.extract.tokenized_chapters.multiprocess ${warning} --processes=12
kubectl auth can-i create pods
"${postprocess_dir}/spark_job.sh" tokenized_chapters ${warning}