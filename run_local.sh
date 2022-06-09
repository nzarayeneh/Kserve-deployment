#!/bin/bash
DOCKER_USER=nzarayeneh
docker run -v $(pwd):/opt/kserve-demo -ePORT=8080 -p8080:8080 ${DOCKER_USER}/kserve-base:latest

# run inference
# curl localhost:8080/v1/models/tg-gcn-kserve-demo:predict -d @./input.json