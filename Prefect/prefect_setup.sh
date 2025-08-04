#!/bin/bash
set -e

echo "setting up prefect api key"

prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

echo "Please create a prefect server in another terminal with: prefect server start"

if ! lsof -i :4200 > /dev/null; then
  echo "Prefect server is not running!"
  echo "Please start it in a separate terminal:"
  echo "prefect server start"
  exit 1
fi

echo "Setting up Prefect"

echo "Creating a work-pool"

prefect work-pool create mlops-25-work-pool --type process --overwrite

echo "Creating a deployment"
prefect deploy Prefect/prefect_flow.py:run --pool mlops-25-work-pool --name mlops-25-deployment 

echo "starting a worker"
prefect worker start --pool mlops-25-work-pool
