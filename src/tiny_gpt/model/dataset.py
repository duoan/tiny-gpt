from torch import from_numpy
from torch.utils.data.dataset import Dataset
from tiny_gpt.model.config import TinyGPTConfig
import numpy as np


class PretrainDataset(Dataset):
    """PretrainDataset"""

    def __init__(self, path: str, config: TinyGPTConfig):
        super(PretrainDataset, self).__init__()
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.len = len(self.data) - config.seq_len
        self.seq_len = config.seq_len

    def __getitem__(self, index):
        x = from_numpy(self.data[index : index + self.seq_len]).long()
        y = from_numpy(self.data[index + 1 : index + 1 + self.seq_len]).long()
        return x, y

    def __len__(self):
        return self.len
