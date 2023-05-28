import torch
from PIL import Image
import numpy as np
import torchvision
from torch.utils.data import TensorDataset, DataLoader


class BrainTumorDataset(TensorDataset):
    """
    Класс для хранения и трансформирования исходных данных
    """
    def __init__(self, x, y, transform='T.Resize((224, 224))'):
        self.transform = transform
        self.x = x
        self.y = y
        self.len = len(y)
    
    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return torch.FloatTensor(np.asarray(self.transform(Image.fromarray(self.x[i])))), torch.LongTensor([self.y[i]])