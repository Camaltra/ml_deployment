from src.utils.pipeline import Pipeline
from src.utils.model.u_net import UNet
from src.utils.dataset import get_train_transform, get_valid_transform, get_loaders
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
import time
from dvclive import Live
from typing import Any
import torchvision
import os


class Trainer(Pipeline):
    def __init__(
        self,
        image_size: int,
        batch_size: int,
        lr: float,
        num_epoch: int,
        augmentations: dict[Any],
        model_save_fpath: str,
        training_tmp_output_base_fpath: str,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_epoch = num_epoch
        self.image_size = image_size
        self.model_save_fpath = model_save_fpath

        self.device = self._get_device()
        self.logger = logging.getLogger("Trainer")
        self.model = UNet(3, 1).to(self.device)

        train_transform = get_train_transform(self.image_size, **augmentations)
        valid_transform = get_valid_transform(self.image_size)

        self.opt = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = nn.BCEWithLogitsLoss()

        self.trn_loader, self.val_loader = get_loaders(train_transform, valid_transform)
        self.tmp_train_result_file_path = f"{training_tmp_output_base_fpath}/{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        if not os.path.exists(self.tmp_train_result_file_path):
            os.makedirs(self.tmp_train_result_file_path)

    @staticmethod
    def _get_device():
        if torch.backends.cuda.is_built():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def run(self):
        self._train()

    def _train(self):
        with Live(save_dvc_exp=True) as live:
            for epoch in range(self.num_epoch):
                train_loss = self._train_fn()

                valid_loss, valid_accuracy, valid_dice_score = (
                    self._compute_val_metrics()
                )

                live.log_metric("train/loss", train_loss)
                live.log_metric("valid/loss", valid_loss)
                live.log_metric("valid/acc", valid_accuracy.item())
                live.log_metric("valid/dice", valid_dice_score.item())

                self._save_samples_predicted(epoch)

                live.next_step()

        torch.save(self.model, self.model_save_fpath)

    def _train_fn(self):
        loop = tqdm(self.trn_loader)
        total_loss = 0

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(self.device)
            targets = targets.float().unsqueeze(1).to(self.device)

            predictions = self.model(data)
            loss = self.criterion(predictions, targets)
            total_loss += loss.item()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loop.set_postfix(loss=loss.item())

        return total_loss / len(self.trn_loader)

    def _compute_val_metrics(self):
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        total_loss = 0

        self.model.eval()
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device).unsqueeze(1)
                preds = self.model(x)
                activated_preds = torch.sigmoid(preds)
                activated_preds = (activated_preds > 0.5).float()
                num_correct += (activated_preds == y).sum()
                num_pixels += torch.numel(activated_preds)
                dice_score += (2 * (activated_preds * y).sum()) / (
                    (activated_preds + y).sum() + 1e-8
                )
                total_loss += self.criterion(preds, y.float()).item()
        self.model.train()
        global_accuracy = num_correct / num_pixels * 100
        global_dice_score = dice_score / len(self.val_loader)
        total_loss /= len(self.val_loader)

        return total_loss, global_accuracy, global_dice_score
    
    def _save_samples_predicted(self, epoch: int):
        self.model.eval()
        for idx, (x, y) in enumerate(self.val_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                preds = torch.sigmoid(self.model(x))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(preds, f"{self.tmp_train_result_file_path}/{epoch}_pred_{idx}.png")
            torchvision.utils.save_image(
                y.unsqueeze(1).float(), f"{self.tmp_train_result_file_path}/{epoch}_truth_{idx}.png"
            )
            break
