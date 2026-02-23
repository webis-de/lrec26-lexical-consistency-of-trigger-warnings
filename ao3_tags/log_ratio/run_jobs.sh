#!/bin/bash
# Meant to be called by ao3_tags/run_jobs.sh
warning=$1
job_id=$2

dir="$(dirname $(readlink -f $0))"

kubectl auth can-i create pods
${dir}/spark_job.sh log_ratios ${warning} ${job_id}