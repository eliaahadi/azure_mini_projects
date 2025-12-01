#!/usr/bin/env python3
"""
day2_blob_io.py

Upload and download a CSV file to/from Azure Blob Storage.
Use env vars:
  AZURE_STORAGE_CONNECTION_STRING
  AZURE_STORAGE_CONTAINER
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient


def get_blob_service_client() -> BlobServiceClient:
    load_dotenv()
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set")
    return BlobServiceClient.from_connection_string(conn_str)


def ensure_container(service_client: BlobServiceClient, container_name: str):
    container_client = service_client.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container()
        print(f"Created container: {container_name}")
    else:
        print(f"Container exists: {container_name}")
    return container_client


def upload_blob(container_client, local_path: Path, blob_name: str):
    blob_client = container_client.get_blob_client(blob_name)
    with open(local_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)
    print(f"Uploaded {local_path} -> {blob_name}")


def download_blob(container_client, blob_name: str, download_path: Path):
    blob_client = container_client.get_blob_client(blob_name)
    with open(download_path, "wb") as f:
        data = blob_client.download_blob().readall()
        f.write(data)
    print(f"Downloaded {blob_name} -> {download_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 2: upload and download CSV with Azure Blob."
    )
    parser.add_argument(
        "--container",
        default=os.getenv("AZURE_STORAGE_CONTAINER", "ml-data"),
        help="Blob container name (defaults to AZURE_STORAGE_CONTAINER or 'ml-data').",
    )
    parser.add_argument(
        "--local-path",
        default="day1_breast_cancer.csv",
        help="Local CSV file to upload.",
    )
    parser.add_argument(
        "--blob-name",
        default="raw/day1_breast_cancer.csv",
        help="Blob name (path) for upload.",
    )
    parser.add_argument(
        "--download-path",
        default="downloaded_day1_breast_cancer.csv",
        help="Where to save downloaded CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    service_client = get_blob_service_client()
    container_client = ensure_container(service_client, args.container)

    local_path = Path(args.local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # Upload
    upload_blob(container_client, local_path, args.blob_name)

    # Download
    download_path = Path(args.download_path)
    download_blob(container_client, args.blob_name, download_path)


if __name__ == "__main__":
    main()