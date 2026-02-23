#!/bin/bash

# Get the necessary parameters from the ./ao3tags/conf/config.yaml file (using the parse_yaml function from utils.sh)
package_dir=$(dirname $(dirname $(readlink -f $0)))
. ${package_dir}/utils/utils.sh --source-only
eval $( parse_yaml ${package_dir}/conf/config.yaml )

TAG="0.0.1-hf-transformers-pytorch"

# Get the directory of this script to reference the Dockerfile
pwd="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
repo_dir="$(dirname $(dirname "$pwd"))"

cd $repo_dir

# Adapt the resource_path to work correctly in the image and remove keys from config
conf_path="${repo_dir}/ao3_tags/conf"
conf_file="${conf_path}/config.yaml"
tmp_conf_file="${conf_path}/tmp_config.yaml"
cp $conf_file $tmp_conf_file

python $conf_path/utils.py $conf_path

# Create the image for the driver (based on a beam image)
docker build  -t ${docker_registry}/${docker_repository}:${TAG} \
              -f ${pwd}/Dockerfile .

# Replace the configuration with the original values
mv $tmp_conf_file $conf_file

# Push the image to the registry
docker login ${docker_registry}
docker push ${docker_registry}/${docker_repository}:${TAG}