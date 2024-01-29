import argparse
from box import ConfigBox

from src.utils.load_params import load_params
from src.utils.train import Trainer


def train(params: ConfigBox) -> None:
    image_size = params.transform.patch_size
    num_epoch = params.train.num_epoch
    model_save_fpath = params.train.model_pickle_fpath
    augmentations = params.train.augmentations
    training_tmp_output_base_fpath = params.train.training_tmp_output_base_fpath
    lr = params.train.lr
    batch_size = params.train.batch_size
    random_state = params.base.random_state

    trainer = Trainer(
        image_size=image_size,
        batch_size=batch_size,
        lr=lr,
        num_epoch=num_epoch,
        augmentations=augmentations,
        model_save_fpath=model_save_fpath,
        training_tmp_output_base_fpath=training_tmp_output_base_fpath,
        random_state=random_state,
    )
    trainer.run()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    train(params)
