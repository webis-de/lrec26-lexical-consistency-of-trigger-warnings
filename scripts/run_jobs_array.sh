#!/bin/bash
job_type=$1

# Example IDs; Put in the IDs you want to process
declare -a arr=("4a48e97b56" "bdf4eac047")

dir="$(dirname $(readlink -f $0))"

# Ask if the PC should be shutdown after completion
read -p "Should the PC be shutdown after completion of all jobs? [y/n] " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
  do_shutdown=1
  echo
  echo
  echo "Will shutdown after completion"
  echo
fi

# Loop over all job ids
for i in "${arr[@]}"; do
  kubectl auth can-i create pods
  "${dir}/${job_type}/run_jobs.sh" abuse "${i}"
  sleep 10
done

# Delete all pods for the current job type that finished successfully
sleep 30
job_type="${job_type/"_"/"-"}"
kubectl -n spark-jobs get pods --field-selector=status.phase==Succeeded \
      | grep "ao3-abuse-${job_type}*" | awk '{print $1}' | xargs kubectl delete pod -n spark-jobs

# Shutdown
if [[ ${do_shutdown} -eq "1" ]]
then
  shutdown -h +5 "Computer will shutdown in 5 minutes"
fi
