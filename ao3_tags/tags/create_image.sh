#!/bin/bash

# Get the necessary parameters from the ./ao3tags/conf/config.yaml file (using the parse_yaml function from utils.sh)
package_dir=$(dirname $(dirname $(readlink -f $0)))
. ${package_dir}/utils/utils.sh --source-only
eval $( parse_yaml ${package_dir}/conf/config.yaml )

SPARK_VERSION=3.4.2
SCALA_VERSION=2.12
PYTHON_VERSION=3.10

TAG="${SPARK_VERSION}-py${PYTHON_VERSION}"

# Get the directory of this script to reference the Dockerfile
pwd="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
repo_dir="$(dirname $(dirname "$pwd"))"

# Get the correct graphframes jar
JAR_SUFFIX="spark${SPARK_VERSION:0:3}-s_${SCALA_VERSION:0:4}"

wget -nc -P ${pwd} \
    "https://repos.spark-packages.org/graphframes/graphframes/0.8.3-${JAR_SUFFIX}/graphframes-0.8.3-${JAR_SUFFIX}.jar"

cd $repo_dir

# Create the image
image_path=${docker_registry}/{docker_repository}:${TAG}-graphframes
docker build  -t ${image_path} \
              --build-arg base_img="${docker_registry}/${docker_parent_repository}:${TAG}" \
              --build-arg jar_name="graphframes-0.8.3-${JAR_SUFFIX}.jar" \
              --build-arg spark_version=${SPARK_VERSION} \
              -f ${pwd}/Dockerfile .

docker push ${image_path}