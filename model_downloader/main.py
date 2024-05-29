import argparse
import os
import shutil
from typing import Any

import yaml
from azure.ai.ml import MLClient
from azure.identity import EnvironmentCredential


def get_config_dict(config_file: str, config_secret_file: str) -> dict[str, Any]:
    with open(config_file, "r") as fp:
        config_dict = yaml.safe_load(fp)
    if os.path.exists(config_secret_file):
        with open(config_secret_file, "r") as fp_secret:
            config_secret_dict = yaml.safe_load(
                fp_secret
            )  # in pipelines these secrets will be stored as env vars
    else:
        config_secret_dict = None
    return config_dict, config_secret_dict


def set_env_vars(config_dict: dict[str, Any], config_secret_dict: dict[str, Any]):
    env_vars = list(config_dict["model_downloader"]["env"].keys())
    for env_var in env_vars:
        os.environ[env_var] = config_dict["model_downloader"]["env"][env_var]

    if config_secret_dict is not None:
        secret_env_vars = list(config_secret_dict["model_downloader"]["env"].keys())
        for secret_var in secret_env_vars:
            os.environ[secret_var] = config_secret_dict["model_downloader"]["env"][
                secret_var
            ]


def download_model(
    ml_client: MLClient,
    model_name_to_download: str,
    model_version_to_download: str,
    model_download_location: str,
):
    ml_client.models.download(
        name=model_name_to_download,
        version=model_version_to_download,
        download_path=model_download_location,
    )


def copy_model_folder(source: str, dest: str):
    shutil.copytree(source, dest, dirs_exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str)
    parser.add_argument("--config-secrets-file", type=str)
    args = parser.parse_args()

    config_file = args.config_file
    config_secrets_file = args.config_secrets_file
    config_dict, config_secrets_dict = get_config_dict(config_file, config_secrets_file)
    set_env_vars(config_dict, config_secrets_dict)
    ml_client = MLClient(
        subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_ML_RESOURCE_GROUP_NAME"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE_NAME"],
        credential=EnvironmentCredential(),
    )
    model_name = config_dict["model_downloader"]["model"]["model_name"]
    model_version = config_dict["model_downloader"]["model"]["version"]
    download_path = config_dict["model_downloader"]["download_params"][
        "model_download_location"
    ]

    download_model(ml_client, model_name, model_version, download_path)

    copy_model_folder(
        os.path.join(download_path, model_name),
        os.path.join("app"),
    )

    shutil.rmtree(download_path)


if __name__ == "__main__":
    main()
