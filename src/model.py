"""Models for training."""
import torch.nn as nn
import torch.nn.functional as F


# model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(in_features=128, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=10)

    def forward(self, out):
        out = F.relu(self.conv1(out))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        out = F.relu(self.conv3(out))
        out = self.pool(out)
        # get the batch size and reshape
        bs, _, _, _ = out.shape
        out = F.adaptive_avg_pool2d(out, 1).reshape(bs, -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class SimpleNetDropNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.dropout1 = nn.Dropout(p=0.3)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, out):
        out = self.bn1(F.max_pool2d(F.relu(self.conv1(out)), (2, 2)))
        out = self.bn2(F.max_pool2d(F.relu(self.conv2(out)), 2))
        out = out.view(out.shape[0], -1)
        out = self.dropout1(self.bn3(F.relu(self.fc1(out))))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out
    