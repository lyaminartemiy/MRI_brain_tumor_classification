import torch
from PIL import Image
import numpy as np
import torchvision
from torch.utils.data import TensorDataset, DataLoader


class BrainTumorDataset(TensorDataset):
    """
    Класс для хранения и трансформирования исходных данных.

    Атрибуты:
        transform (str или torchvision.transforms.Transform): Преобразование, применяемое к изображениям.
            По умолчанию используется 'T.Resize((224, 224))'.
        x (list или numpy.ndarray): Входные данные (изображения).
        y (list или numpy.ndarray): Целевые метки.
    
    Методы:
        __len__(): Возвращает длину набора данных.
        __getitem__(i): Возвращает i-ый элемент набора данных.

    """

    def __init__(self, x, y, transform='T.Resize((224, 224))'):
        """
        Инициализация класса BrainTumorDataset.

        Аргументы:
            x (list или numpy.ndarray): Входные данные (изображения).
            y (list или numpy.ndarray): Целевые метки.
            transform (str или torchvision.transforms.Transform): Преобразование, применяемое к изображениям.
                По умолчанию используется 'T.Resize((224, 224))'.
        """

        self.transform = transform
        self.x = x
        self.y = y
        self.len = len(y)
    
    def __len__(self):
        """
        Возвращает длину набора данных.

        Возвращает:
            int: Длина набора данных.
        """
        return self.len

    def __getitem__(self, i):
        """
        Возвращает i-ый элемент набора данных.

        Аргументы:
            i (int): Индекс элемента.

        Возвращает:
            tuple: Кортеж, содержащий преобразованное изображение и соответствующую целевую метку.
        """
        return (
            torch.FloatTensor(np.asarray(self.transform(Image.fromarray(self.x[i])))),
            torch.LongTensor([self.y[i]])
        )
