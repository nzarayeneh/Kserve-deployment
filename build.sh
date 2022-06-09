#!/bin/bash

set -e

if [ -z "$1" ]; then
    version="latest"
else
    version=$1
fi

#echo "---- Collecting artifacts ----"
## Tutorials
#if [ -d "mlworkbench-docs" ]; then
#    cd mlworkbench-docs;
#    git pull
#    cd ..
#else
#    git clone https://github.com/TigerGraph-DevLabs/mlworkbench-docs.git
#fi


# Build the base Jupyter image
echo ---- Building image ---- 
docker build --rm -t kserve-base:${version} .

echo ---- Pushing to docker hub ----
# replace "skirpichenko" with "tigergraphml"
DOCKER_USER=nzarayeneh
docker tag kserve-base:${version} ${DOCKER_USER}/kserve-base:${version}
docker push ${DOCKER_USER}/kserve-base:${version}

#echo ---- Clean up ----
#rm -rf mlworkbench-docs
echo ${DOCKER_USER}