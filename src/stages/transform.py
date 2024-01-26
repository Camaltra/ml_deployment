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

    train_img_dir_path = data_dir / params.transform.train_img_dir_path
    train_mask_dir_path = data_dir / params.transform.train_mask_dir_path
    test_img_dir_path = data_dir / params.transform.test_img_dir_path
    test_mask_dir_path = data_dir / params.transform.test_mask_dir_path

    create_folders([train_img_dir_path, train_mask_dir_path, test_img_dir_path, test_mask_dir_path])

    valid_idx_set = set(params.transform.valid_idx)
    patch_size = params.transform.patch_size

    builder = BuildDataset(
        raw_data_path,
        train_img_dir_path,
        train_mask_dir_path,
        test_img_dir_path,
        test_mask_dir_path,
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
