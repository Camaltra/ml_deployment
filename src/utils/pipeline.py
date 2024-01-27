from abc import ABC, abstractmethod
import torch


class Pipeline(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def _get_device():
        if torch.backends.cuda.is_built():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
