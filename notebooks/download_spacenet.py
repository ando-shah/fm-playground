#!/usr/bin/env python3
"""
Download SpaceNet Dataset from AWS S3

This script demonstrates how to download the SpaceNet dataset from AWS S3 using Python's boto3 library
instead of the AWS CLI, with the ability to specify a custom download location.
"""

import boto3
import os
import sys
import threading
from botocore import UNSIGNED
from botocore.client import Config

# Progress tracking class
class ProgressPercentage(object):
    def __init__(self, client, bucket, key):
        self._size = client.head_object(Bucket=bucket, Key=key)['ContentLength']
        self._seen_so_far = 0
        self._lock = threading.Lock()
        
    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(f"\rDownload progress: {percentage:.2f}% ({self._seen_so_far}/{self._size} bytes)")
            sys.stdout.flush()

def download_spacenet(download_dir='.', show_progress=True):
    """
    Download SpaceNet dataset to a specified directory
    
    Args:
        download_dir (str): Directory where the file will be downloaded
        show_progress (bool): Whether to show download progress
    
    Returns:
        str: Path to the downloaded file
    """
    # Create an S3 client with unsigned configuration (for public datasets)
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # Define the bucket and object key
    bucket_name = 'spacenet-dataset'
    object_key = 'spacenet/SN1_buildings/tarballs/SN1_buildings_train_AOI_1_Rio_3band.tar.gz'
    filename = 'SN1_buildings_train_AOI_1_Rio_3band.tar.gz'
    
    # Create the directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Define the full local file path where the dataset will be saved
    local_file_path = os.path.join(download_dir, filename)
    
    print(f"Downloading {object_key} from {bucket_name}...")
    print(f"File will be saved to: {os.path.abspath(local_file_path)}")
    
    # Download the file with or without progress tracking
    if show_progress:
        s3_client.download_file(
            bucket_name, 
            object_key, 
            local_file_path,
            Callback=ProgressPercentage(s3_client, bucket_name, object_key)
        )
        print("\n")  # Add a newline after progress bar
    else:
        s3_client.download_file(bucket_name, object_key, local_file_path)
    
    print(f"Download complete! File saved to {os.path.abspath(local_file_path)}")
    return local_file_path

if __name__ == "__main__":
    # Example 1: Download to current directory
    # download_spacenet()
    
    # Example 2: Download to a relative path
    download_spacenet('data/spacenet')
    
    # Example 3: Download to an absolute path
    # download_spacenet('/home/username/datasets/spacenet')
    
    # Example 4: Download to user's home directory
    # download_spacenet(os.path.join(os.path.expanduser('~'), 'datasets/spacenet')) 