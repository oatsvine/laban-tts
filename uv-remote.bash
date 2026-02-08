#!/bin/bash

REMOTE_HOST="server3"
REMOTE_SRC_DIR="/home/vmadmin/src/oatsvine/projects/laban-tts"
echo 'Starting rsync to server3...'
rsync -avP --exclude .git --exclude .venv --exclude .env . ${REMOTE_HOST}:${REMOTE_SRC_DIR}/

# Check if arguments are provided, if not exit.
if [ "$#" -eq 0 ]; then
    echo "No arguments provided. Exiting."
    exit 1
fi

ssh -t ${REMOTE_HOST} "cd ${REMOTE_SRC_DIR} && docker compose run -e UV_ENV_FILE=.env --rm --entrypoint uv tts  $(printf '%q ' "$@")"
