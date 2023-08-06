"""Functions for training and validation."""
import os
from collections.abc import Callable
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

matplotlib.style.use('ggplot')


def seed_everything(seed: int):
    """
    Make default settings for random values.

    Args:
        seed (int): seed for random
    """
    import os
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    # будет работать - если граф вычислений не будет меняться во время обучения
    torch.backends.cudnn.benchmark = True  # оптимизации


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    scheduler: Callable,
    dataloader: torch.utils.data.DataLoader,
    device,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model (nn.Module): current model
        optimizer (torch.optim.Optimizer): optimizer for this learning
        criterion (Callable): criterion for this learning
        scheduler (Callable): scheduler for this learning
        dataloader (torch.utils.data.DataLoader): loader for this learning
        device: set 'cpu' or 'cuda'

    Returns:
        Tuple[float, float]: mean loss and mean accuracy
    """
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    total_epoch_steps = int(len(dataloader.dataset)/dataloader.batch_size)

    for _, batch in tqdm(enumerate(dataloader), total=total_epoch_steps):
        [images, target] = batch
        images, target = images.to(device), target.to(device)
        outputs = model(images)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    train_loss = train_running_loss/len(dataloader.dataset)
    train_accuracy = 100.0 * train_running_correct/len(dataloader.dataset)    
    return train_loss, train_accuracy


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Callable,
    device,
) -> Tuple[float, float]:
    """
    Validate model for current params.

    Args:
        model: current model
        dataloader: loader for validation
        criterion: loss function for this learning
        device: set 'cpu' or 'cuda'

    Returns:
        loss and accuracy for this dataloader
    """
    model.eval()

    val_running_loss: float = 0
    val_running_correct = 0

    with torch.no_grad():
        inference_steps = int(len(dataloader.dataset)/dataloader.batch_size)

        for _, batch in tqdm(enumerate(dataloader), total=inference_steps):            
            [images, target] = batch
            images = images.to(device)
            target = target.to(device)
            outputs = model(images)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()
        val_loss = val_running_loss/len(dataloader.dataset)
        val_accuracy = 100. * val_running_correct/len(dataloader.dataset)
        return val_loss, val_accuracy


def model_learning(
    epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    scheduler,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device,
) -> Tuple[dict, dict]:
    """
    Make learning of model for epochs.

    Args:
        epochs: number of epochs
        model: current model
        optimizer: optimizer for this learning
        criterion: loss function for this learning
        scheduler: scheduler for changing optimizer
        train_loader: loader for train model
        test_loader: loader for validate model
        device: set 'cpu' or 'cuda'

    Returns: 
        dicts with losses and accuracies
    """
    loss = {'train': [], 'val': []}
    accuracy = {'train': [], 'val': []}
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1} of {epochs}')
        train_epoch_loss, train_epoch_accuracy = train_one_epoch(model,
                                                                 optimizer,
                                                                 criterion,
                                                                 scheduler,
                                                                 train_loader,
                                                                 device)

        loss['train'].append(train_epoch_loss)
        accuracy['train'].append(train_epoch_accuracy)

        print(f'\nTrain Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}')

        val_loss, val_accuracy = validate(model, test_loader, criterion, device)
        loss['val'].append(val_loss)
        accuracy['val'].append(val_accuracy)
        print(f'\nTest Loss: {val_loss:.4f}, Test Acc: {val_accuracy:.2f}')
    return loss, accuracy


def plot_training(
    loss: dict,
    accuracy: dict,
):
    """
    Create and save plots of losses and accuracy.

    Args:
        loss: list of losses
        accuracy: list of accuracy
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 1, 1)
    plt.xlabel('epoch')
    plt.plot(loss['train'], label='train_loss')
    plt.plot(loss['val'], label='valid_loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.xlabel('epoch')
    plt.plot(accuracy['train'], label='train accuracy')
    plt.plot(accuracy['val'], label='valid accuracy')
    plt.legend()
    os.mkdir('../outputs')
    plt.savefig('../outputs/initial_training_loss.png')


def calc_accuracy(
    model: nn.Module, 
    loader: torch.utils.data.DataLoader,
    device,
) -> float:
    """
    Calculate accuracy for loader

    Args:
        model: model
        loader: data_loader
        device: 'cpu' or 'cuda'
    Returns:
        accuracy for this model
    """

    test_acc: float = 0
    model.eval()
    for samples, labels in loader:
        with torch.no_grad():
            samples, labels = samples.to(device), labels.to(device)
            output = model(samples)
            pred = torch.argmax(output, dim=1)
            correct = pred.eq(labels)
            test_acc += torch.mean(correct.float())
    return round(test_acc.item()*100.0/len(loader), 2)
