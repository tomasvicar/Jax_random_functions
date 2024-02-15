#!/bin/bash

# Get the current directory name, which is assumed to be the code folder
CODE_DIR_NAME=$(basename "$(pwd)" | tr '[:upper:]' '[:lower:]')

# Move one directory up from the current directory
PARENT_DIR=$(dirname "$(pwd)")

# Set Docker image and container names based on the code directory name
DOCKER_IMAGE_TAG="${CODE_DIR_NAME}_image"
CONTAINER_NAME="${CODE_DIR_NAME}_cont"

# Stop and remove the existing container if it exists
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Remove the existing Docker image if it exists
docker image rm $DOCKER_IMAGE_TAG 2>/dev/null || true

# Build the new Docker image
docker build -t $DOCKER_IMAGE_TAG .

