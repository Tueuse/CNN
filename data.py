from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # 内置CIFAR-10数据集

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪（带填充，避免边缘信息丢失）
    transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10官方均值
                         std=[0.2470, 0.2435, 0.2616])   # CIFAR-10官方标准差
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

def get_data():
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    return train_loader, test_loader

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')