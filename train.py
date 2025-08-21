# 训练函数
import numpy as np
import torch
from pyarrow.dataset import dataset
from tqdm import tqdm


def train(model, train_loader, criterion, optimizer, epochs, device):
    train_loss_path = []
    train_acc_path = []
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}") as loop:
            for images, labels in train_loader:
                # 数据迁移到设备
                images, labels = images.to(device), labels.to(device)
                # 清零梯度
                optimizer.zero_grad()
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)
                # 反向传播+参数更新
                loss.backward()
                optimizer.step()
                # 统计损失与准确率
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)  # 取概率最大的类别
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # 更新进度条
                loop.update(1)
                loop.set_postfix(loss=running_loss / len(train_loader), acc=100 * correct / total)

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            loop.set_postfix(train_loss=train_loss, train_acc=train_acc)
            train_loss_path.append(train_loss)
            train_acc_path.append(train_acc)

    torch.save(model.state_dict(), './model.pth')
    return train_loss_path, train_acc_path


# 测试函数
def test(model, test_loader, criterion, device):
    confusion_matrix = np.zeros((10, 10), dtype=int)
    model = model.to(device)
    model.eval()  # 设为评估模式（关闭BN、Dropout）
    test_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(test_loader, total=len(test_loader), desc="Epoch test")

    with torch.no_grad():  # 禁用梯度计算（节省内存，加速）
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            labels_np = labels.cpu().numpy()
            preds_np = predicted.cpu().numpy()
            for t, p in zip(labels_np, preds_np):
                confusion_matrix[t, p] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.update(1)
            loop.set_postfix(loss=test_loss / len(test_loader), acc=100 * correct / total)

    # 计算测试集指标
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    loop.set_postfix(test_loss=test_loss, test_acc=test_acc)
    return test_loss, test_acc, confusion_matrix