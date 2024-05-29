#!/bin/sh
echo "downloading the model"
python -m model_downloader.main --config-file config.yaml --config-secrets-file config_secret.yaml

echo "building the image"
az acr build --resource-group diabetes-classifier-2024 --registry diabetesclassifier2024 --image diabetes-classifier-api:latest ./app
