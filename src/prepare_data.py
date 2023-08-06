"""Load datasets with transforms."""
from torchvision import datasets
from torchvision.transforms import transforms

# define transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),],)
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),],)

# train and validation data
train_data = datasets.CIFAR10(
    root='../input/data',
    train=True,
    download=True,
    transform=transform_train,
)
val_data = datasets.CIFAR10(
    root='../input/data',
    train=False,
    download=True,
    transform=transform_val,
)
