import os

import tarfile
import zipfile

def untar_file(file_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    if file_path.endswith("tar.gz"):
        method = "r:gz"
    elif file_path.endswith("tar"):
        method = "r:"
        
    with tarfile.open(file_path, method) as tar:
        tar.extractall(save_path)

def unzip_file(file_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)