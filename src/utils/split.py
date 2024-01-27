from src.utils.pipeline import Pipeline
import os
import shutil
import random


class Splitter(Pipeline):
    def __init__(
        self,
        valid_dir_path_img: str,
        valid_dir_path_mask: str,
        train_dir_path_img: str,
        train_dir_path_mask: str,
        prc_valid: int,
        random: int,
    ):
        super().__init__()
        self.valid_dir_path_img = valid_dir_path_img
        self.valid_dir_path_mask = valid_dir_path_mask
        self.train_dir_path_img = train_dir_path_img
        self.train_dir_path_mask = train_dir_path_mask
        self.prc_valid = prc_valid
        self.random = random

    def run(self):
        random.seed(self.random)
        train_data = os.listdir(self.train_dir_path_img)
        num_to_select = int(len(train_data) * self.prc_valid)
        selected_images = random.sample(train_data, num_to_select)

        for filename in selected_images:
            src_img = os.path.join(self.train_dir_path_img, filename)
            src_mask = os.path.join(self.train_dir_path_mask, filename)

            dest_img = os.path.join(self.valid_dir_path_img, filename)
            dest_mask = os.path.join(self.valid_dir_path_mask, filename)

            shutil.move(src_img, dest_img)
            shutil.move(src_mask, dest_mask)
