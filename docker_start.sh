#!/bin/bash

# Get the current directory name, which is assumed to be the code folder
CODE_DIR_NAME_LOWER=$(basename "$(pwd)" | tr '[:upper:]' '[:lower:]')
CODE_DIR_NAME=$(basename "$(pwd)")

# Move one directory up from the current directory
PARENT_DIR=$(dirname "$(pwd)")

# Set Docker image and container names based on the code directory name
DOCKER_IMAGE_TAG="${CODE_DIR_NAME_LOWER}_image"
CONTAINER_NAME="${CODE_DIR_NAME_LOWER}_cont"

# Set the notebook directory path
NOTEBOOK_DIR="/workspace/$CODE_DIR_NAME"

# Check if the container with the same name already exists
if [ $(docker ps -a -q -f name=^/${CONTAINER_NAME}$) ]; then
    # Check if the container is running and stop it if it is
    if [ $(docker ps -q -f name=^/${CONTAINER_NAME}$) ]; then
        echo "Stopping container ${CONTAINER_NAME}."
        docker stop $CONTAINER_NAME
    fi
    # Remove the container
    echo "Removing container ${CONTAINER_NAME}."
    docker rm $CONTAINER_NAME
fi

# Create and start a new container
# --ipc=host and --shm-size=1g are about shared memory for multiprocessing
echo "Creating and starting a new container ${CONTAINER_NAME}."
echo "CODE_DIR_NAME: $CODE_DIR_NAME"
docker run -it \
  --gpus all \
  --ipc=host \
  --shm-size=1g \
  --name $CONTAINER_NAME \
  -v $PARENT_DIR:/workspace \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e NOTEBOOK_DIR=$NOTEBOOK_DIR \
  $DOCKER_IMAGE_TAG \
  bash
