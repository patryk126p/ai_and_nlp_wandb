"""
Download dataset
"""
import argparse
import logging
import os
import pickle

import torchvision
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms as T

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="download", save_code=True)
    run.config.update(args)
    logger.info(f"Downloading {args.dataset}")
    dataset_class = getattr(torchvision.datasets, args.dataset)
    transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_set = dataset_class(
        root="data", train=True, download=True, transform=transforms
    )
    test_set = dataset_class(
        root="data", train=False, download=True, transform=transforms
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    data = (train_loader, test_loader)

    file_path = os.path.join("data", args.artifact_name)
    with open(file_path, "wb") as fh:
        pickle.dump(data, fh)

    logger.info(f"Uploading {args.artifact_name} to artifact store")
    artifact = wandb.Artifact(
        args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(file_path)
    run.log_artifact(artifact)
    artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download torchvision dataset")
    parser.add_argument(
        "dataset", type=str, help="Name of torchvision class for downloading dataset"
    )
    parser.add_argument("batch_size", type=int, help="Batch size for DataLoader")
    parser.add_argument("artifact_name", type=str, help="Dataset name")
    parser.add_argument("artifact_type", type=str, help="Dataset type")
    parser.add_argument("artifact_description", type=str, help="Dataset description")
    args = parser.parse_args()
    go(args)
