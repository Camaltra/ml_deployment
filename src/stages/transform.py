import sys
from pathlib import Path

import argparse
import os

from src.utils.load_params import load_params
from src.utils.transform import BuildDataset


def create_folders(folder_paths: list[Path]):
    for folder_path in folder_paths:
        if not folder_path.exists():
            os.makedirs(folder_path)


def transform(params):
    data_dir = Path(params.transform.raw_dataset_dir)
    raw_data_path = data_dir / params.transform.raw_dataset_fname

    train_dir_path_img = data_dir / params.transform.train_dir_path / "img"
    train_dir_path_mask = data_dir / params.transform.train_dir_path / "mask"
    valid_dir_path_img = data_dir / params.transform.valid_dir_path / "img"
    valid_dir_path_mask = data_dir / params.transform.valid_dir_path / "mask"

    create_folders([train_dir_path_img, train_dir_path_mask, valid_dir_path_img, valid_dir_path_mask])

    valid_idx_set = set(params.transform.valid_idx)
    patch_size = params.transform.patch_size

    builder = BuildDataset(
        raw_data_path,
        train_dir_path_img,
        train_dir_path_mask,
        valid_dir_path_img,
        valid_dir_path_mask,
        valid_idx_set,
        patch_size
    )
    builder.run()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    transform(params)
