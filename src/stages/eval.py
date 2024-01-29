from pathlib import Path

import argparse
from box import ConfigBox

from src.utils.load_params import load_params
from src.utils.eval import Evaluator


def evaluation(params: ConfigBox) -> None:
    data_dir = Path(params.transform.raw_dataset_dir)
    model_save_fpath = params.train.model_pickle_fpath
    test_dir_path_img = data_dir / params.transform.test_dir_path / "img"
    test_dir_path_mask = data_dir / params.transform.test_dir_path / "mask"

    metrics_fpath = Path(params.eval.metrics_file)
    metrics_fpath.parent.mkdir(parents=True, exist_ok=True)

    patch_size = params.transform.patch_size

    evaluator = Evaluator(
        model_save_fpath,
        test_dir_path_img,
        test_dir_path_mask,
        patch_size,
        metrics_fpath,
    )
    evaluator.run()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    evaluation(params)
