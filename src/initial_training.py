import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import utils as utils
from model import SimpleNetDropNorm
from prepare_data import train_data, val_data


def train_model(
    batch_size: int = 16,
    epochs: int = 1,
    lr: float = 0.001,
    seed: int = 42,
    device=torch.device('cpu'),
):
    """
    Build all together: create data loaders, initialize the model,
    optimizer and loss function, train model,
    create loss and accuracy plots, save model.

    Args:
        batch_size (int): set batch size
        epochs (int): number of epochs
        lr (float): learning rate
        seed (int): seed for randoms
        device : set "cpu" or "cuda"
    """
    utils.seed_everything(seed)

    # train data loader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
    )

    # initialize the model, optimizer and loss function
    model = SimpleNetDropNorm().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), 
                           eps=1e-8, weight_decay=0.0005
                           )
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # train
    loss, accuracy = utils.model_learning(epochs, model, optimizer, criterion, scheduler, train_loader, val_loader, device)

    # loss and accuracy plots
    utils.plot_training(loss, accuracy)
    # save model checkpoint
    torch.save({'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'scheduler': scheduler.state_dict(),
                }, '../outputs/model.pth',)

    accuracy = utils.calc_accuracy(model, val_loader, device)
    print('End. Final accuracy', accuracy)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(batch_size=132, epochs=2, lr=0.001, seed=42, device=device)
