#!/bin/bash
job_name=$1
warning=$2

# Parse any additional, job specific parameters by their position
shift 2
declare -a additionalParams=("$@")

# Get the necessary parameters from the ./ao3tags/conf/config.yaml file (using the parse_yaml function from utils.sh)
package_dir=$(dirname $(dirname $(dirname $(readlink -f $0))))
. ${package_dir}/utils/utils.sh --source-only
eval $( parse_yaml ${package_dir}/conf/config.yaml )

# Get the data_path from ao3_tags
PATHS=$(python3 ${package_dir}/__init__.py)
IFS=$"|" read -r resource_path data_path<<< ${PATHS}

# Submit the job
spark-submit \
  --deploy-mode cluster \
	--name ao3_${warning}_${job_name} \
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
	${package_dir}/extract/postprocess/${job_name}.py \
	  ${warning} \
	  /data \
	  "${additionalParams[@]}"