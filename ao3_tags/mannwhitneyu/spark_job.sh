#!/bin/bash
warning=$1
job_id=$2

# Get the necessary parameters from the ./ao3tags/conf/config.yaml file (using the parse_yaml function from utils.sh)
package_dir=$(dirname $(dirname $(readlink -f $0)))
. ${package_dir}/utils/utils.sh --source-only
eval $( parse_yaml ${package_dir}/conf/config.yaml )

# Get the data_path from ao3_tags
PATHS=$(python3 ${package_dir}/__init__.py)
IFS=$"|" read -r resource_path data_path<<< ${PATHS}

# Create the output directory if it does not exist
mkdir -p ${data_path}/mannwhitneyu/z_scores/${warning}

# Submit the job
spark-submit \
  --deploy-mode cluster \
	--name ao3_${warning}_mannwhitneyu \
	--driver-memory $spark_base_driver_memory \
	--executor-memory $spark_base_executor_memory \
	--conf spark.dynamicAllocation.enabled="true" \
  --conf spark.dynamicAllocation.shuffleTracking.enabled="true" \
  --conf spark.dynamicAllocation.initialExecutors=$spark_base_initial_executors \
  --conf spark.dynamicAllocation.maxExecutors=$spark_base_max_executors \
	--conf spark.kubernetes.container.image=$spark_base_image_url \
	--conf spark.kubernetes.driver.volumes.hostPath.data.options.path=$data_path \
	--conf spark.kubernetes.executor.volumes.hostPath.data.options.path=$data_path \
	--conf spark.kubernetes.container.image.pullPolicy=Always \
	--conf spark.kubernetes.driver.volumes.hostPath.data.mount.path=/data \
	--conf spark.kubernetes.executor.volumes.hostPath.data.mount.path=/data \
	${package_dir}/mannwhitneyu/z_scores.py \
	  ${warning} \
	  /data \
	  ${job_id}