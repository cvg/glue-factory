import argparse
import os
import random
from pathlib import Path
from stat import S_ISDIR

import paramiko
from tqdm import tqdm

HOST = "student-cluster.inf.ethz.ch"
HOST_OUTPUT_ROOT = "/cluster/courses/3dv/data/team-2"


def list_files(sftp, remote_dir, file_ending):
    files = []
    for entry in sftp.listdir_attr(remote_dir):
        remote_path = os.path.join(remote_dir, entry.filename)
        if S_ISDIR(entry.st_mode):  # Check if it's a directory
            files += list_files(sftp, remote_path, file_ending)
        elif remote_path.endswith(file_ending):
            files.append(remote_path)
    return files


def sample_and_download_files(
    ssh_client, remote_dir, local_dir, file_ending, num_samples
):
    # Get a list of all image files in the remote directory and its subdirectories
    sftp = ssh_client.open_sftp()
    print("Searching for files in " + remote_dir)
    files = list_files(sftp, remote_dir, file_ending)
    print("Found " + str(len(files)) + " files")

    # Randomly select some image files
    print(f"Select and download from {remote_dir} to {local_dir}...")
    selected_files = random.sample(files, num_samples)

    # Download the selected files
    for file in tqdm(selected_files):
        remote_file_path = Path(file)
        local_path = (
            Path(local_dir) / remote_file_path.parent.name / remote_file_path.name
        )
        if not local_path.parent.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(file, local_path)

    # Close the connections
    sftp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Output folder to store sampled files in",
        required=True,
    )
    parser.add_argument(
        "--remote_folder",
        type=str,
        help="remote folder to look for files to sample (as subfolder of /cluster/courses/3dv/data/team-2)",
        required=True,
    )
    parser.add_argument("--user", type=str, help="username on cluster", required=True)
    parser.add_argument(
        "--password", type=str, help="password for user on cluster", required=True
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="number of samples to download"
    )
    args = parser.parse_args()

    # SSH connection details
    username = str(args.user)
    password = str(args.password)

    # Connect to the remote host
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=HOST, username=username, password=password)

    # Set the remote directory containing images
    remote_dir = os.path.join(HOST_OUTPUT_ROOT, args.remote_folder)
    # Set the local directory to download images
    local_dir = os.path.join("/", args.output_folder)

    # Create the local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    sample_and_download_files(
        client, remote_dir, local_dir, file_ending=".hdf5", num_samples=args.num_samples
    )
    client.close()
