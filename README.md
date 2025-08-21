# CNN
CNN model to classifies CIFAR-10 dataset with accuracy of 82.5%
# 基于CNN神经网络模型对CIFAR-10数据集进行分类

## 实验背景

数据集：CIFAR-10

神经网络：CNN

代码生成：豆包

## 代码模块

```python
model.py	# 模型定义
data.py		# 数据加载
train.py	# 模型训练测试
run.py		# 主程序
```

## 模型结构

```
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)            
        x = self.relu(self.conv3(x))
        x = self.pool(x)            
        x = self.relu(self.conv4(x))
        x = self.pool(x)            


        x = x.view(-1, 128 * 4 * 4)

        x = self.dropout(self.relu(self.fc1(x)))

        x = self.fc2(x)

        return x
```

## 训练结果

```cmd
Epoch 1: 100%|██████████| 782/782 [00:23<00:00, 33.04it/s, train_acc=37.5, train_loss=1.69]
Epoch 2: 100%|██████████| 782/782 [00:23<00:00, 33.81it/s, train_acc=55.5, train_loss=1.25]
Epoch 3: 100%|██████████| 782/782 [00:23<00:00, 33.14it/s, train_acc=63.6, train_loss=1.04]
Epoch 4: 100%|██████████| 782/782 [00:24<00:00, 32.19it/s, train_acc=67.7, train_loss=0.928]
Epoch 5: 100%|██████████| 782/782 [00:24<00:00, 31.62it/s, train_acc=70.9, train_loss=0.855]
Epoch 6: 100%|██████████| 782/782 [00:24<00:00, 31.61it/s, train_acc=73.1, train_loss=0.79]
Epoch 7: 100%|██████████| 782/782 [00:23<00:00, 32.59it/s, train_acc=74.5, train_loss=0.745]
Epoch 8: 100%|██████████| 782/782 [00:24<00:00, 31.92it/s, train_acc=75.9, train_loss=0.712]
Epoch 9: 100%|██████████| 782/782 [00:23<00:00, 33.01it/s, train_acc=77.2, train_loss=0.682]
Epoch 10: 100%|██████████| 782/782 [00:23<00:00, 33.33it/s, train_acc=77.9, train_loss=0.655]
Epoch 11: 100%|██████████| 782/782 [00:24<00:00, 32.46it/s, train_acc=78.4, train_loss=0.639]
Epoch 12: 100%|██████████| 782/782 [00:24<00:00, 31.61it/s, train_acc=79.1, train_loss=0.62]
Epoch 13: 100%|██████████| 782/782 [00:24<00:00, 31.87it/s, train_acc=79.6, train_loss=0.599]
Epoch 14: 100%|██████████| 782/782 [00:23<00:00, 32.87it/s, train_acc=80.3, train_loss=0.584]
Epoch 15: 100%|██████████| 782/782 [00:24<00:00, 32.07it/s, train_acc=81.1, train_loss=0.564]
Epoch 16: 100%|██████████| 782/782 [00:24<00:00, 31.59it/s, train_acc=81, train_loss=0.565]
Epoch 17: 100%|██████████| 782/782 [00:24<00:00, 31.49it/s, train_acc=81.8, train_loss=0.542]
Epoch 18: 100%|██████████| 782/782 [00:23<00:00, 32.95it/s, train_acc=82.2, train_loss=0.537]
Epoch 19: 100%|██████████| 782/782 [00:24<00:00, 31.95it/s, train_acc=82.4, train_loss=0.525]
Epoch 20: 100%|██████████| 782/782 [00:23<00:00, 33.11it/s, train_acc=82.8, train_loss=0.509]
Epoch test: 100%|██████████| 157/157 [00:16<00:00,  9.30it/s, test_acc=82.5, test_loss=0.533]
```

![image-20250821163727951](C:\Users\Jarwww\Desktop\项目\CNN\imgs\image-20250821163727951.png)


![image-20250821163812461](C:\Users\Jarwww\Desktop\项目\CNN\imgs\image-20250821163812461.png)
