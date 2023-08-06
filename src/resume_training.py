import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import utils
from model import CNN
from prepare_data import train_data, val_data


def check_resume_training():
    """ Check resume training when save and load model. """
    # learning parameters
    batch_size = 128
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.seed_everything(42)

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
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), 
                            eps=1e-8, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epochs_first = 1
    utils.model_learning(epochs_first, 
                        model, 
                        optimizer, 
                        criterion, 
                        scheduler, 
                        train_loader, 
                        val_loader, 
                        device)


    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'scheduler': scheduler.state_dict(),
                }, '../outputs/model_test_resume_training.pth')

    # load model
    checkpoint = torch.load('../outputs/model_test_resume_training.pth')
    model_second = CNN().to(device)

    model_second.load_state_dict(checkpoint['model_state_dict'])
    optimizer_second = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), 
                            eps=1e-8, weight_decay=0.0005)
    optimizer_second.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion_second = checkpoint['loss']
    scheduler_second = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler_second.load_state_dict(checkpoint['scheduler'])

    # check that models are with same accuracy
    accuracy = utils.calc_accuracy(model, val_loader, device)
    print('Final accuracy 1: ', accuracy)

    accuracy_second = utils.calc_accuracy(model_second, val_loader, device)
    print('Final accuracy 2: ', accuracy_second)

    # checking parameters
    all_is_good = True
    for a, b in zip(model.parameters(), model_second.parameters()):
        if torch.sum(a != b) != torch.Tensor([0]):
            all_is_good = False
    print('After training: ')
    print('Models are the same' if all_is_good else 'Different models')


if __name__ == '__main__':
    check_resume_training()
