import torch
from matplotlib import pyplot as plt
from torch import nn, optim
import seaborn as sns
import data
import train
import model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 20

    train_dataloader, test_dataloader = data.get_data()
    model = model.get_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # static_dict = torch.load('model.pth', weights_only=True)
    # model.load_state_dict(static_dict)

    train_loss_path, train_acc_path = train.train(model, train_dataloader, criterion, optimizer, epochs, device)
    test_loss, test_acc, confusion_matrix= train.test(model, test_dataloader, criterion, device)

    plt.figure(figsize=(10, 10))  # 设置图大小（根据类别数量调整）
    # 绘制热力图：annot=True显示单元格数值，fmt='d'表示整数格式
    sns.heatmap(
        confusion_matrix,
        annot=True,  # 显示单元格内的样本数量
        fmt='d',  # 数值格式（整数）
        cmap='Blues',  # 颜色映射（浅蓝→深蓝，数值越大颜色越深）
        xticklabels=data.classes,  # x轴：预测标签
        yticklabels=data.classes,  # y轴：真实标签
        cbar_kws={'label': 'data_num'}  # 颜色条标签
    )
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # 子图1：损失曲线（同上）
    ax1.plot(range(1, epochs + 1), train_loss_path, label='train_loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：准确率曲线（同上）
    ax2.plot(range(1, epochs + 1), train_acc_path, label='train_acc', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.show()