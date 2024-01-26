from abc import ABC, abstractmethod


class Pipeline(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass
