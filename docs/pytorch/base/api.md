# PyTorch 神经网络 API

## Pytorch 神经网络基础

### 层和块

在之前的内容中，我们认识了一些神经网络，比如：线性回归，Softmax 回归，多层感知机；他们有的是整个模型，有的是一层神经网络，有的甚至只是一个单元，他们的功能以及复杂程度也各不相同，但他们都有着如下三个特征：

- 接受一些输入
- 产生对应的输出
- 由一组可调整参数描述

对于一些复杂的网络，研究讨论比层大但比整个模型小的部分很有意义，因为复杂的网络中经常有重复出现的部分，每个部分也常常有自己的功能。考虑到上面的三个特征，这就使得我们思考是否可以对这些部分进行一个抽象，这就得到了块的概念：块指单个层，多个层组成的部分，或者整个模型本身。使用块对整个模型进行描述就简便许多，这一过程是递归的，块的内部还可以划分为多个块，直至满足需要为止。

PyTorch 帮我们实现了块的大部分所需功能，包括自动求导，我们只需从 nn.Module 继承并改写其中的一部分就能得到我们需要的块以及模型，具体做法和细节见代码中的注释

### 参数管理

在选择了架构并设置了超参数后，我们就进入了训练阶段。此时，我们的目标是找到使损失函数最小化的模型参数值。经过训练后，我们将需要使用这些参数来做出未来的预测。此外，有时我们希望提取参数，以便在其他环境中复用它们，将模型保存下来，以便它可以在其他软件中执行，或者为了获得科学的理解而进行检查。

此部分主要为代码实现，笔记见代码中的注释

### 延后初始化

有时在建立网络时，我们不会指定网络的输入输出维度，也就不能确定网络的参数形状，深度学习框架支持延后初始化，即当第一次将数据传入模型时自动的得到所有的维度，然后初始化所有的参数。

PyTorch 也支持这一点，比如 nn.LazyLinear，但本门课程中并未介绍。

### 自定义层

深度学习成功背后的一个因素是神经网络的灵活性：我们可以用创造性的方式组合不同的层，从而设计出适用于各种任务的架构。例如，研究人员发明了专门用于处理图像、文本、序列数据和执行动态规划的层。同样的，对于层而言，深度学习框架并不能满足我们所有的需求，然而，层本身也具有极大的灵活性，我们可以自定义想要的层。

此部分主要为代码实现，笔记见代码中的注释

### 读写文件

到目前为止，我们讨论了如何处理数据，以及如何构建、训练和测试深度学习模型。然而，有时我们希望保存训练的模型，以备将来在各种环境中使用（比如在部署中进行预测）。此外，当运行一个耗时较长的训练过程时，最佳的做法是定期保存中间结果，以确保在服务器电源被不小心断掉时，我们不会损失几天的计算结果。

## API

### 层和块

`nn.Sequential` 定义了一种特殊的 Module。

```python
# 回顾一下多层感知机
import torch
from torch import nn

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)
print(net(X))
```

某一次的输出结果：

```shell
tensor([[-0.0834,  0.0413,  0.3388, -0.1317,  0.2923, -0.0273,  0.1292, -0.0553,
          0.2576, -0.1454],
        [ 0.0371,  0.0283,  0.3756, -0.3132,  0.1236,  0.1512,  0.0881,  0.0118,
          0.3466, -0.0126]], grad_fn=<AddmmBackward0>)
```

### 自定义块

```python
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类的__init__函数
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


# 实例化多层感知机的层，然后在每次调用正向传播函数调用这些层
net = MLP()
X = torch.rand(2, 20)
print(net(X))
```

某一次的输出结果：

```shell
tensor([[-0.0535, -0.1295, -0.1194,  0.2260,  0.2259,  0.0245, -0.0291, -0.1338,
         -0.1508, -0.1677],
        [ 0.0133, -0.0673,  0.0188,  0.0655,  0.2237,  0.0261, -0.0431, -0.2538,
         -0.2707, -0.0861]], grad_fn=<AddmmBackward0>)
```

### 顺序块

```python
import torch
from torch import nn


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block  # block 本身作为它的key，存在_modules里面的为层，以字典的形式

    def forward(self, X):
        for block in self._modules.values():
            print(block)
            X = block(X)
        return X


net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)
print(net(X))
```

某一次的输出结果：

```shell
Linear(in_features=20, out_features=256, bias=True)
ReLU()
Linear(in_features=256, out_features=10, bias=True)
tensor([[ 0.0380, -0.1235,  0.0134, -0.0737,  0.1052, -0.2154, -0.0231,  0.1517,
         -0.1856, -0.2350],
        [ 0.0520, -0.0822, -0.1645, -0.1138,  0.2141, -0.0826, -0.0697,  0.1139,
         -0.1674, -0.1338]], grad_fn=<AddmmBackward0>)
```

### 正向传播

```python
import torch
from torch import nn
import torch.nn.functional as F


# 在正向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight + 1))
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


net = FixedHiddenMLP()
X = torch.rand(2, 20)
print(net(X))
```

某一次的输出结果：

```shell
tensor(0.3551, grad_fn=<SumBackward0>)
```

### 混合组合块

```python
import torch
from torch import nn
import torch.nn.functional as F


# 在正向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight + 1))
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


# 混合代培各种组合块的方法
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


chimear = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
X = torch.rand(2, 20)
print(chimear(X))
```

某一次的输出结果：

```shell
tensor(-0.0127, grad_fn=<SumBackward0>)
```

### 参数管理

```python
# 首先关注具有单隐藏层的多层感知机
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))
print(net[2].state_dict())  # 访问参数，net[2]就是最后一个输出层
print(type(net[2].bias))  # 目标参数
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad == None)  # 还没进行反向计算，所以grad为None
print(*[(name, param.shape) for name, param in net[0].named_parameters()])  # 一次性访问所有参数
print(*[(name, param.shape) for name, param in net.named_parameters()])  # 0是第一层名字，1是ReLU，它没有参数
print(net.state_dict()['2.bias'].data)  # 通过名字获取参数
```

某一次的输出结果：

```shell
tensor([[-0.3903],
        [-0.3722]], grad_fn=<AddmmBackward0>)
OrderedDict([('weight', tensor([[-0.3075, -0.1058, -0.0616,  0.1174, -0.1603,  0.2533, -0.0329,  0.0310]])), ('bias', tensor([-0.2481]))])
<class 'torch.nn.parameter.Parameter'>
Parameter containing:
tensor([-0.2481], requires_grad=True)
tensor([-0.2481])
True
('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
tensor([-0.2481])
```

### 嵌套块

```python
import torch
import torch.nn as nn


# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())  # f'block{i}' 可以传一个字符串名字过来，block2可以嵌套四个block1
    return net


X = torch.rand(size=(2, 4))
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
print(rgnet)
```

某一次的输出结果：

```shell
tensor([[-0.4293],
        [-0.4293]], grad_fn=<AddmmBackward0>)
Sequential(
  (0): Sequential(
    (block0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True)
)
```

### 内置初始化

```python
import torch.nn as nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 下划线表示把m.weight的值替换掉
        nn.init.zeros_(m.bias)


net.apply(init_normal)  # 会递归调用 直到所有层都初始化
print(net[0].weight.data[0])
print(net[0].bias.data[0])
```

某一次的输出结果：

```shell
tensor([-0.0133, -0.0207, -0.0111,  0.0138])
tensor(0.)
```

### 参数替换

```python
import torch.nn as nn


# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])  # 打印名字是啥，形状是啥
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5  # 这里*=的代码相当于先计算一个布尔矩阵(先判断>=)，然后再用布尔矩阵的对应元素去乘以原始矩阵的每个元素。保留绝对值大于5的权重，不是的话就设为0


net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
net.apply(my_init)
print(net[0].weight[:2])
net[0].weight.data[:] += 1  # 参数替换
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])
```

某一次的输出结果：

```shell
Init weight torch.Size([8, 4])
Init weight torch.Size([1, 8])
tensor([[ 0.0000, -0.0000,  0.0000, -0.0000],
        [-6.5086,  7.6410,  5.1407, -9.8181]], grad_fn=<SliceBackward0>)
tensor([42.,  1.,  1.,  1.])
```

### 自定义层

```python
# 构造一个没有任何参数的自定义层
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

# 将层作为组件合并到构建更复杂的模型中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean())

# 带参数的图层


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))  # nn.Parameter使得这些参数加上了梯度
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


dense = MyLinear(5, 3)
print(dense.weight)

# 使用自定义层直接执行正向传播计算
print(dense(torch.rand(2, 5)))
# 使用自定义层构建模型
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))
```

某一次的输出结果：

```shell
tensor([-2., -1.,  0.,  1.,  2.])
tensor(3.7253e-09, grad_fn=<MeanBackward0>)
Parameter containing:
tensor([[ 0.9900,  1.4626,  0.2577],
        [-1.4476, -0.1842,  1.4564],
        [ 1.2600,  2.0309,  0.9797],
        [-0.3708, -0.3384, -0.3367],
        [-0.1536,  1.5189,  1.1400]], requires_grad=True)
tensor([[0.9266, 2.5753, 3.0402],
        [1.0008, 1.2875, 1.8410]])
tensor([[0.],
        [0.]])
```
