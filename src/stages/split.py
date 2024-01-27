import sys
from pathlib import Path

import argparse
import os

from src.utils.load_params import load_params
from src.utils.split import Splitter
from src.utils.folder_utils import create_folders


def split(params):
    data_dir = Path(params.transform.raw_dataset_dir)
    valid_dir_path_img = data_dir / params.split.valid_dir_path / "img"
    valid_dir_path_mask = data_dir / params.split.valid_dir_path / "mask"
    train_dir_path_img = data_dir / params.transform.train_dir_path / "img"
    train_dir_path_mask = data_dir / params.transform.train_dir_path / "mask"
    random = params.base.random_state

    create_folders([valid_dir_path_img, valid_dir_path_mask])

    prc_valid = params.split.prc_valid

    splitter = Splitter(
        valid_dir_path_img,
        valid_dir_path_mask,
        train_dir_path_img,
        train_dir_path_mask,
        prc_valid,
        random,
    )
    splitter.run()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    split(params)
