import os
import subprocess
import boto3
import argparse
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    if path is None:
        return None

    iter_tracker_file, wandb_tracker_file  = get_checkpoint_tracker_filename(path)
    # if not os.path.exists(tracker_file):
    #     print("Checkpoint tracker file does not exist: %s" % tracker_file)
    #     return None

    if check_s3_path_exists(iter_tracker_file):
        # sync the tracker file from S3 to local directory
        print("Syncing tracker file from S3 to local directory...")
        subprocess.run(["aws", "s3", "cp", iter_tracker_file, local_checkpoint_dir], check=True)
        subprocess.run(["aws", "s3", "cp", wandb_tracker_file, local_checkpoint_dir], check=True)
    else:
        print("Tracker file does not exist in S3: %s" % iter_tracker_file)
        return None

    with open(os.path.join(local_checkpoint_dir, "latest_checkpointed_iteration.txt"), "rb") as f:
        iteration = int(f.read().decode())
    ckpt_path = os.path.join(path, directory_format.format(iteration))
    if not check_s3_path_exists(ckpt_path):
        print("Checkpoint does not exist: %s" % ckpt_path)
        return None

    print("Found checkpoint: %s" % ckpt_path)
    return ckpt_path,iteration


def get_checkpoint_tracker_filename(root_path: str):
    """
    Tracker file records the latest checkpoint during training to restart from.
    """
    return os.path.join(root_path, "latest_checkpointed_iteration.txt"),os.path.join(root_path, "latest_checkpointed_wandb_run_id.txt")


def check_s3_path_exists(s3_path):
    """
    Check if the given S3 path exists.
    """
    try:
        s3 = boto3.client("s3")
        bucket_name, prefix = s3_path.replace("s3://", "").split("/", 1)
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        return "Contents" in response  # If the key exists, 'Contents' will be in the response
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"Error: {str(e)}")
        return False


def sync_s3_to_local(s3_path, local_path):
    """
    Sync the S3 directory to a local directory using the AWS CLI.
    """
    try:
        subprocess.run(["aws", "s3", "sync", s3_path, local_path], check=True)
        print(f"Successfully synced {s3_path} to {local_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error syncing S3 directory: {str(e)}")

def get_specific_checkpoint(s3_checkpoint_dir,local_checkpoint_dir,iteration):
    """
    Sync the specific checkpoint from S3 to local directory.
    """
    # Construct the S3 path for the specific checkpoint
    s3_checkpoint_path = os.path.join(s3_checkpoint_dir, "global_step_{}".format(iteration))
    
    # Check if the S3 path exists
    if check_s3_path_exists(s3_checkpoint_path):
        print(f"S3 path exists: {s3_checkpoint_path}")
        
        # Sync the checkpoint folder to the local directory
        sync_s3_to_local(s3_checkpoint_path, os.path.join(local_checkpoint_dir, "global_step_{}".format(iteration)))
    else:
        print(f"S3 path does not exist: {s3_checkpoint_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Sync S3 checkpoint directory to local directory.")
    parser.add_argument("--s3_checkpoint_dir", type=str, required=True, help="S3 checkpoint directory path")
    parser.add_argument("--local_checkpoint_dir", type=str, required=True, help="Local directory to sync checkpoints")

    args = parser.parse_args()
    s3_checkpoint_dir = args.s3_checkpoint_dir
    local_checkpoint_dir = args.local_checkpoint_dir

    # Check if the S3 path exists
    if check_s3_path_exists(s3_checkpoint_dir):
        print(f"S3 path exists: {s3_checkpoint_dir}")
        
        # Use find_latest_ckpt_path to get the checkpoint path
        latest_ckpt_path,iteration = find_latest_ckpt_path(s3_checkpoint_dir)
        if latest_ckpt_path:
            print(f"Latest checkpoint path: {latest_ckpt_path}")
            
            # Sync the checkpoint folder to the local directory
            sync_s3_to_local(latest_ckpt_path, os.path.join(local_checkpoint_dir, "global_step_{}".format(iteration)))
        else:
            print("No valid checkpoint found.")
    else:
        print(f"S3 path does not exist: {s3_checkpoint_dir}")