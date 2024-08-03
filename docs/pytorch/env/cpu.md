# CPU 版 Pytorch 搭建

CPU 版本的 Pytorch 可以直接安装，不需要 GPU 支持。

## 安装

打开 Anaconda Prompt，切换至新建好的环境，输入以下命令安装 CPU 版 Pytorch：

```shell
conda install pytorch torchvision torchaudio cpuonly
```

等待安装结束即可。

![](../images/env/cpu1.png)

## 验证

### 验证安装成功

输入列表命令查看安装的包：

```shell
conda list
```

若含有 `torch` 或 `pytorch`，则说明安装成功。

![](../images/env/cpu2.png)

### 验证版本正确

新建 Python 脚本（注意环境要正确），输入以下代码：

```python
import torch
print(torch.cuda.is_available())
```

运行脚本，如果输出 `False`，则说明版本正确。

## Hello, Pytorch!

至此，我们已经在本地配置好了 pytorch 的运行环境，我们可以用一个非常简单的例子来测试一下。

新建 Python 文件，输入以下代码并运行：

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 定义模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    transform=transform,
    download=True
)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
model = SimpleNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# 训练函数
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# 测试函数
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 训练和测试循环
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

本实验是一个经典的图像分类任务，使用 MNIST 数据集。

某一次的输出结果：

```shell
...download...
Epoch 1
-------------------------------
loss: 2.317173  [    0/60000]
loss: 0.278135  [ 6400/60000]
loss: 0.126566  [12800/60000]
loss: 0.203645  [19200/60000]
loss: 0.058915  [25600/60000]
loss: 0.160201  [32000/60000]
loss: 0.065284  [38400/60000]
loss: 0.153437  [44800/60000]
loss: 0.262129  [51200/60000]
loss: 0.064724  [57600/60000]
Test Error:
 Accuracy: 96.1%, Avg loss: 0.115400

Epoch 2
-------------------------------
loss: 0.019986  [    0/60000]
loss: 0.045924  [ 6400/60000]
loss: 0.120131  [12800/60000]
loss: 0.032515  [19200/60000]
loss: 0.174265  [25600/60000]
loss: 0.023037  [32000/60000]
loss: 0.023797  [38400/60000]
loss: 0.022360  [44800/60000]
loss: 0.046270  [51200/60000]
loss: 0.073648  [57600/60000]
Test Error:
 Accuracy: 97.1%, Avg loss: 0.096175

Epoch 3
-------------------------------
loss: 0.007136  [    0/60000]
loss: 0.008195  [ 6400/60000]
loss: 0.005358  [12800/60000]
loss: 0.016153  [19200/60000]
loss: 0.011466  [25600/60000]
loss: 0.030158  [32000/60000]
loss: 0.101332  [38400/60000]
loss: 0.070195  [44800/60000]
loss: 0.039363  [51200/60000]
loss: 0.045191  [57600/60000]
Test Error:
 Accuracy: 97.9%, Avg loss: 0.071707

Epoch 4
-------------------------------
loss: 0.032217  [    0/60000]
loss: 0.021705  [ 6400/60000]
loss: 0.137437  [12800/60000]
loss: 0.007645  [19200/60000]
loss: 0.010579  [25600/60000]
loss: 0.059706  [32000/60000]
loss: 0.076431  [38400/60000]
loss: 0.022402  [44800/60000]
loss: 0.050549  [51200/60000]
loss: 0.078512  [57600/60000]
Test Error:
 Accuracy: 97.3%, Avg loss: 0.093361

Epoch 5
-------------------------------
loss: 0.015435  [    0/60000]
loss: 0.036592  [ 6400/60000]
loss: 0.002897  [12800/60000]
loss: 0.003369  [19200/60000]
loss: 0.031028  [25600/60000]
loss: 0.050782  [32000/60000]
loss: 0.004176  [38400/60000]
loss: 0.002142  [44800/60000]
loss: 0.022591  [51200/60000]
loss: 0.009988  [57600/60000]
Test Error:
 Accuracy: 97.7%, Avg loss: 0.089208

Done!
```

如果报错，可以尝试依次运行：

```shell
conda uninstall numpy
conda install numpy==1.26.4
```
