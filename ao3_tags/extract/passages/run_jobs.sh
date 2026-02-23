#!/bin/bash
# Meant to be called by ao3_tags/run_jobs.sh
warning=$1
job_id=$2

postprocess_dir="$(dirname $(dirname $(readlink -f $0)))/postprocess"

python -m ao3_tags.extract.passages.sample_words ${warning} ${job_id}
python -m ao3_tags.extract.passages.identify_chapters ${warning} ${job_id}
python -m ao3_tags.extract.passages.retrieve_chapters ${warning} ${job_id}
python -m ao3_tags.extract.passages.identify_passages ${warning} ${job_id}