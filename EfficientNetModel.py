import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader

from efficientnet_pytorch import EfficientNet

import numpy as np

from tqdm import trange
from time import sleep


class EfficientNetModel:
    def __init__(self, version, path_to_model_dict=None):
        
        assert version == 'efficientnet-b1', "The version should be 'efficientnet-b1'"
        
        # Создание модели версии - version
        self.model = EfficientNet.from_pretrained(version)
        
        # заменяем последний слой
        self.model._fc = nn.Linear(1280, 3)
        self.model._fc.requires_grad_ = True
        
        # История обучения и тестирования
        self.history = None
        
        # Загрузка весов модели, если такая существует
        if path_to_model_dict is None:
            next
        else:
            self.model.load_state_dict(torch.load(path_to_model_dict))
            
        # установка доступного устройства
        self.DEVICE = torch.device("cpu")
        
    def load_model(self, path_to_model_dict):
        """
        Метод для загрузки весов обученной модели
        
        Аргументы:
        path_to_model_dict -- str путь до файла модели
        """
        self.model.load_state_dict(torch.load(path_to_model_dict))
        
    def save_model(self, name_model_dict):
        """
        Метод для сохранения весов обученной модели
        
        Аргументы:
        name_model_dict -- str название файла для сохранения
        """
        torch.save(self.model.state_dict(), name_model_dict)
        
    def train(self, max_epochs, optimizer, loss_fn, train_dataloader, val_dataloader):
        """
        Метод для обучения нейросети

        Аргументы:
        max_epochs -- количество эпох int
        optimizer -- torch.optim оптимизатор
        loss_fn -- функция потерь

        Возвращает:
        словарь со значениями функции потерь в процессе обучения
        """

        if not torch.cuda.is_available():
            print('CUDA is not available.  Training on CPU ...')
        else:
            print('CUDA is available!  Training on GPU ...')
        
        self.history = {
            'losses': {'train': [], 'valid': []},
            'accuracy': {'train': [], 'valid': []}
        }

        for epoch in trange(max_epochs, desc="Progress Bar", unit="carrots"):

            train_losses_per_epoch = []

            running_corrects = 0
            processed_data = 0

            # Модуль обучения
            self.model.train()

            for X_batch, y_batch in train_dataloader:

                X_batch, y_batch = X_batch.to(self.DEVICE), y_batch.to(self.DEVICE).type(torch.LongTensor)
                y_batch = y_batch.view(1, -1)[0]

                # Обновление градиентов
                optimizer.zero_grad()

                # Выход модели
                output = self.model(X_batch.float())

                # Подсчёт значения функции потерь
                loss = loss_fn(output, y_batch)

                # Подсчёт градиентов
                loss.backward()

                # Шаг оптимизатора
                optimizer.step()

                train_losses_per_epoch.append(loss.item())
                
                preds = torch.argmax(output, 1)
                running_corrects += torch.sum(preds == y_batch.data)
                processed_data += X_batch.size(0)

            self.history['losses']['train'].append(np.mean(train_losses_per_epoch))
            self.history['accuracy']['train'].append((running_corrects / processed_data).item())

            # Модуль тестирования
            self.model.eval()

            valid_losses_per_epoch = []

            running_corrects = 0
            processed_data = 0

            for X_batch, y_batch in val_dataloader:

                X_batch, y_batch = X_batch.to(self.DEVICE), y_batch.to(self.DEVICE).type(torch.LongTensor)
                y_batch = y_batch.view(1, -1)[0]

                with torch.no_grad():

                    # Выход модели
                    output = self.model(X_batch.float())

                    # Подсчёт значения функции потерь
                    loss = loss_fn(output, y_batch)

                valid_losses_per_epoch.append(loss.item())
                
                preds = torch.argmax(output, 1)
                running_corrects += torch.sum(preds == y_batch.data)
                processed_data += X_batch.size(0)

            self.history['losses']['valid'].append(np.mean(valid_losses_per_epoch))
            self.history['accuracy']['valid'].append((running_corrects / processed_data).item())

        return self.history
    
    def freeze_weights(self, number_freezing):
        """
        Метод для заморозки весов слоёв нейросети
        
        Аргументы:
        number_freezing -- число замороженных слоёв
        """
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = bool(i > number_freezing)
            
    def predict(self, data_loader):
        """
        Метод для предсказывания классов на тестовой выборке
        
        Аргументы:
        data -- torch.utils.data.dataloader даталоадер для теста
        """
        # Режим тестирования
        self.model.eval()
        
        with torch.no_grad():
            
            logits = []
            
            for X_batch in data_loader:

                    X_batch = X_batch.to(self.DEVICE)

                    # Выход модели
                    output = self.model(X_batch.float()).cpu()

                    logits.append(outputs)
                    
        probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
        return probs
    
    def predict_item(self, mini_batch):
        """
        Метод для предсказывания класса изображения
        
        Аргументы:
        data -- torch.utils.data.dataloader даталоадер для теста
        """
        # Режим тестирования
        self.model.eval()
        
        with torch.no_grad():
            
            # Выход модели
            output = self.model(mini_batch.float()).cpu()
        
        prob = nn.functional.softmax(output).numpy()
        return prob
