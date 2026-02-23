#!/bin/bash
root_tag=$1
min_distance=$2
max_distance=$3

# Get the necessary parameters from the ./ao3tags/conf/config.yaml file (using the parse_yaml function from utils.sh)
package_dir=$(dirname $(dirname $(readlink -f $0)))
. ${package_dir}/utils/utils.sh --source-only
eval $( parse_yaml ${package_dir}/conf/config.yaml )

# Get the data_path from ao3_tags
PATHS=$(python3 ${package_dir}/__init__.py)
IFS=$"|" read -r resource_path data_path<<< ${PATHS}

# Create the output directory if it does not exist
mkdir -p ${data_path}/tags/

# Submit the job
spark-submit \
  --deploy-mode cluster \
	--name ao3_tag_graph_${root_tag} \
	--driver-memory $spark_graphframes_driver_memory \
	--executor-memory $spark_graphframes_executor_memory \
	--conf spark.dynamicAllocation.enabled="true" \
  --conf spark.dynamicAllocation.shuffleTracking.enabled="true" \
  --conf spark.dynamicAllocation.initialExecutors=$spark_graphframes_initial_executors \
  --conf spark.dynamicAllocation.maxExecutors=$spark_graphframes_max_executors \
	--conf spark.kubernetes.container.image=$spark_graphframes_image_url \
	--conf spark.kubernetes.container.image.pullPolicy=Always \
	--conf spark.kubernetes.driver.volumes.hostPath.data.options.path=$data_path \
	--conf spark.kubernetes.executor.volumes.hostPath.data.options.path=$data_path \
	--conf spark.kubernetes.driver.volumes.hostPath.data.mount.path=/data \
	--conf spark.kubernetes.executor.volumes.hostPath.data.mount.path=/data \
	--conf spark.kubernetes.driver.volumes.hostPath.tags.options.path=$tag_path \
	--conf spark.kubernetes.executor.volumes.hostPath.tags.options.path=$tag_path \
	--conf spark.kubernetes.driver.volumes.hostPath.tags.mount.path=/tags \
	--conf spark.kubernetes.executor.volumes.hostPath.tags.mount.path=/tags \
	${package_dir}/tags/tag_graph.py \
	  ${root_tag} \
	  ${min_distance} \
	  ${max_distance} \
	  /data \
	  /tags

# Aggregate the results
python3 -m ao3_tags.tags.tag_graph_postprocessing ${root_tag}