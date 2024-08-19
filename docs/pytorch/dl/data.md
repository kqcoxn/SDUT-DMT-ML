# 数据处理 API

我们在机器学习中使用过 numpy 与 pandas 进行数据处理，而 PyTorch 在其基础上进行了进一步的封装，提供了数据处理的 API。

## 张量数据操作

张量是数学和物理中表示多维数据的一种通用结构，扩展了标量、向量和矩阵的概念，可以在多维空间中进行线性代数运算。

在 PyTorch 中，张量（Tensor）可以理解为一个 N 维数组，是构建神经网络的基本数据结构。它类似于 NumPy 的 ndarray，但具有更强的功能，特别是对 GPU 的支持，使得大规模的计算更高效。张量可以表示标量（0 维）、向量（1 维）、矩阵（2 维）以及更高维度的数据结构，并支持自动微分以便于神经网络的训练。

### 张量创建

```python
# 导入torch库，应该导入torch，而不是pytorch
import torch

x = torch.arange(12)  # 初始化一个0-11的张量
print(x)
```

输出结果：

```shell
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
```

### 访问张量形状

可以通过张量的 shape 属性来访问张量的形状和张量中元素的总数。

```python
import torch

x = torch.arange(12)  # 初始化一个0-11的张量

print(x.shape)  # 张量的形状
print(x.numel())  # 张量中元素的总数
```

输出结果：

```shell
torch.Size([12])
12
```

### 改变张量形状

要改变一个张量的形状而不改变元素数量和元素值，可以调用 reshape 函数。

```python
import torch

x = torch.arange(12)  # 初始化一个0-11的张量

x = x.reshape(3, 4)  # 一维张量改为3行四列的张量
print(x)
```

输出结果：

```shell
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
```

### 创建全 0、全 1 张量

使用全 0、全 1、其他常量或者从特定分布中随即采样的数字。

```python
import torch

y = torch.zeros((2, 3, 4))
print(y)

y = torch.ones((2, 3, 4))
print(y)
```

输出结果：

```shell
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]])

tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]])
```

### 创建特定值张量

通过提供包含数值的 Python 列表（或嵌套列表）来为所需张量中的每个元素赋予确定值。

```python
import torch

y = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])  # 二维tensor
print(y)
print(y.shape)

z = torch.tensor([[[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]])  # 三维tensor
print(z)
print(z.shape)
```

输出结果：

```shell
tensor([[2, 1, 4, 3],
        [1, 2, 3, 4],
        [4, 3, 2, 1]])
torch.Size([3, 4])

tensor([[[2, 1, 4, 3],
         [1, 2, 3, 4],
         [4, 3, 2, 1]]])
torch.Size([1, 3, 4])
```

### 张量运算操作

常见的标准算术运算符(`+`、`-`、`*`、`/`、和 `**`)都可以被升级为按元素运算。

```python
import torch

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x**y)  # **运算符是求幂运算
```

输出结果：

```shell
tensor([ 3.,  4.,  6., 10.])
tensor([-1.,  0.,  2.,  6.])
tensor([ 2.,  4.,  8., 16.])
tensor([0.5000, 1.0000, 2.0000, 4.0000])
tensor([ 1.,  4., 16., 64.])
```

对每个元素应用更多的计算。

```python
import torch

x = torch.tensor([1.0, 2, 4, 8])

x = torch.exp(x)  # e的x次方
print(x)
```

输出结果：

```shell
tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])
```

### 张量合并

可以把多个张量结合在一起。

```python
import torch

x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

m = torch.cat((x, y), dim=0)  # 按行合并起来
print(m)

n = torch.cat((x, y), dim=1)  # 按列合并起来
print(n)
```

输出结果：

```shell
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [ 2.,  1.,  4.,  3.],
        [ 1.,  2.,  3.,  4.],
        [ 4.,  3.,  2.,  1.]])

tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
        [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
        [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]])
```

### 张量逻辑运算

通过逻辑运算符构建二元张量。

```python
import torch

x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(x)
print(y)

print(x == y)  # 对应元素相等为 True，否则为 False
```

输出结果：

```shell
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])

tensor([[2., 1., 4., 3.],
        [1., 2., 3., 4.],
        [4., 3., 2., 1.]])

tensor([[False,  True, False,  True],
        [False, False, False, False],
        [False, False, False, False]])
```

### 张量累加运算

对张量中所有元素进行求和会产生一个只有一个元素的张量。

```python
import torch

x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(x)

print(x.sum())
```

输出结果：

```shell
tensor(66.)
```

### 张量广播运算

即使形状不同，仍然可以通过调用广播机制(broadcasting mechanism)来执行按元素操作。

```python
import torch

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)

print(a + b)  # a会复制出一个3*2的矩阵，b复制出一个3*2的矩阵，然后再相加，会得到一个3*2矩阵
```

输出结果：

```shell
tensor([[0],
        [1],
        [2]])

tensor([[0, 1]])

tensor([[0, 1],
        [1, 2],
        [2, 3]])
```

### 张量访问运算

与切片语法相同，例如可以用\[-1\]选择最后一个元素，可以用\[1:3\]选择第二个和第三个元素：

```python
import torch

x = torch.arange(12, dtype=torch.float32).reshape((3, 4))

print(x[-1])
print(x[1:3])
```

输出结果：

```shell
tensor([ 8.,  9., 10., 11.])

tensor([[ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])
```

### 张量元素改写

除读取外，还可以通过指定索引来将元素写入矩阵。

```python
import torch

x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(x)

x[1, 2] = 9
print(x)
```

输出结果：

```shell
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])

tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  9.,  7.],
        [ 8.,  9., 10., 11.]])
```

也可以为多个元素赋值相同的值，只需要索引所有元素，然后为它们赋值。

```python
import torch

x = torch.arange(12, dtype=torch.float32).reshape((3, 4))

x[0:2, :] = 12
print(x)
```

输出结果：

```shell
tensor([[12., 12., 12., 12.],
        [12., 12., 12., 12.],
        [ 8.,  9., 10., 11.]])
```

### \[选读\]张量内存变化

运行一些操作可能会导致为新结果分配内容。

```python
import torch

x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

before = id(y)  # 记录y的id

y = x + y
print(id(y) == before)  # 运行操作后，赋值后的y的id和原来的id不一样
```

输出结果：

```shell
False
```

如果在后续计算中没有重复使用 X，即内存不会过多复制，也可以使用 X\[:\] = X + Y 或 X += Y 来减少操作的内存开销。

```python
import torch

x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

before = id(x)

x += y
print(id(x) == before)
```

输出结果：

```shell
True
```

### 张量转 Numpy

```python
import torch

x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(x)

A = x.numpy()
B = torch.tensor(A)
print(type(A), type(B))
```

输出结果：

```shell
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])
<class 'numpy.ndarray'> <class 'torch.Tensor'>
```

将大小为 1 的张量转为 Python 标量:

```python
import torch

a = torch.tensor([3.5])
print(a)
print(a.item())

print(float(a))
print(int(a))
```

输出结果：

```shell
tensor([3.5000])
3.5
3.5
3
```

## 矩阵操作

pytorch 会将张量视为矩阵，因此很多操作都可以直接使用。

### 创建矩阵

```python
import torch

A = torch.arange(20).reshape(5, 4)
print(A)
```

输出结果：

```shell
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]])
```

### 矩阵转置

矩阵的转置。

```python
import torch

A = torch.arange(20).reshape(5, 4)
print(A.T)  # 矩阵的转置
```

输出结果：

```shell
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]])

tensor([[ 0,  4,  8, 12, 16],
        [ 1,  5,  9, 13, 17],
        [ 2,  6, 10, 14, 18],
        [ 3,  7, 11, 15, 19]])
```

### 矩阵克隆（深拷贝）

给定具有相同形状的任何两个张量，任何按元素二元运算的结果都将是相同形状的张量。

```python
import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)

B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print(A + B)
```

输出结果：

```shell
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.],
        [16., 17., 18., 19.]])

tensor([[ 0.,  2.,  4.,  6.],
        [ 8., 10., 12., 14.],
        [16., 18., 20., 22.],
        [24., 26., 28., 30.],
        [32., 34., 36., 38.]])
```

### 矩阵相乘（对应元素相乘）

两个句子的按元素乘法称为哈达玛积（Hadamard product）（数学符号 ⊙）

```python
import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)

print(A * A)
```

输出结果：

```shell
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.],
        [16., 17., 18., 19.]])

tensor([[  0.,   1.,   4.,   9.],
        [ 16.,  25.,  36.,  49.],
        [ 64.,  81., 100., 121.],
        [144., 169., 196., 225.],
        [256., 289., 324., 361.]])
```

### 矩阵加标量

```python
import torch

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(X)

print(a + X)
print((a * X).shape)
```

输出结果：

```shell
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])

tensor([[[ 2,  3,  4,  5],
         [ 6,  7,  8,  9],
         [10, 11, 12, 13]],

        [[14, 15, 16, 17],
         [18, 19, 20, 21],
         [22, 23, 24, 25]]])
torch.Size([2, 3, 4])
```

### 矩阵求和

对于一维向量：

```python
import torch

X = torch.arange(4, dtype=torch.float32)
print(X)

print(X.sum())
```

输出结果：

```shell
tensor([0., 1., 2., 3.])
tensor(6.)
```

对于多维矩阵，可以表示任意形状张量的元素和。

```python
import torch

A = torch.arange(20 * 2).reshape(2, 5, 4)
print(A)

print(A.shape)
print(A.sum())
```

输出结果：

```shell
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11],
         [12, 13, 14, 15],
         [16, 17, 18, 19]],

        [[20, 21, 22, 23],
         [24, 25, 26, 27],
         [28, 29, 30, 31],
         [32, 33, 34, 35],
         [36, 37, 38, 39]]])

torch.Size([2, 5, 4])
tensor(780)
```

也可以指定张量沿哪一个轴来通过求和降低维度：

```python
import torch

A = torch.arange(20 * 2).reshape(2, 5, 4)
print(A)

# 在index==0的维度上求和
A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0)
print(A_sum_axis0.shape)

# 在index==1的维度上求和
A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1)
print(A_sum_axis1.shape)

# 在index==2的维度上求和
A_sum_axis2 = A.sum(axis=2)
print(A_sum_axis2)
print(A_sum_axis2.shape)
```

输出结果：

```shell
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11],
         [12, 13, 14, 15],
         [16, 17, 18, 19]],

        [[20, 21, 22, 23],
         [24, 25, 26, 27],
         [28, 29, 30, 31],
         [32, 33, 34, 35],
         [36, 37, 38, 39]]])

tensor([[20, 22, 24, 26],
        [28, 30, 32, 34],
        [36, 38, 40, 42],
        [44, 46, 48, 50],
        [52, 54, 56, 58]])
torch.Size([5, 4])

tensor([[ 40,  45,  50,  55],
        [140, 145, 150, 155]])
torch.Size([2, 4])

tensor([[  6,  22,  38,  54,  70],
        [ 86, 102, 118, 134, 150]])
torch.Size([2, 5])
```

如果不希望维度丢失，可以用 `sum(axis, keepdims=True)` 来保留维度。

### 矩阵平均值

一个与求和相关的量是平均值（mean 或 average）。

```python
import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)

print(A.mean())  # 平均值
```

输出结果：

```shell
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.],
        [16., 17., 18., 19.]])
tensor(9.5000)
```

### 矩阵广播

```python
import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)

sum_A = A.sum(axis=1, keepdims=True)  # keepdims=True不丢掉维度，否则三维矩阵按一个维度求和就会变为二维矩阵，二维矩阵若按一个维度求和就会变为一维向量
print(sum_A)

print(A / sum_A)
```

输出结果：

```shell
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.],
        [16., 17., 18., 19.]])

tensor([[ 6.],
        [22.],
        [38.],
        [54.],
        [70.]])

tensor([[0.0000, 0.1667, 0.3333, 0.5000],
        [0.1818, 0.2273, 0.2727, 0.3182],
        [0.2105, 0.2368, 0.2632, 0.2895],
        [0.2222, 0.2407, 0.2593, 0.2778],
        [0.2286, 0.2429, 0.2571, 0.2714]])
```

### 按轴累加

沿某个轴计算 A 元素的累加总和。

```python
import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)

print(A.cumsum(axis=0))
```

输出结果：

```shell
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.],
        [16., 17., 18., 19.]])

tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  6.,  8., 10.],
        [12., 15., 18., 21.],
        [24., 28., 32., 36.],
        [40., 45., 50., 55.]])
```

### 向量点积

```python
import torch

x = torch.arange(4, dtype=torch.float32)
print(x)
y = torch.ones(4, dtype=torch.float32)
print(y)

print(torch.dot(x, y))
```

输出结果：

```shell
tensor([0., 1., 2., 3.])
tensor([1., 1., 1., 1.])
tensor(6.)
```

### 矩阵相乘（线性代数相乘）

```python
import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)

B = torch.ones(4, 3)
print(B)

print(torch.mm(A, B))
```

输出结果：

```shell
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.],
        [16., 17., 18., 19.]])

tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])

tensor([[ 6.,  6.,  6.],
        [22., 22., 22.],
        [38., 38., 38.],
        [54., 54., 54.],
        [70., 70., 70.]])
```

### 矩阵向量积

```python
import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)
print(A.shape)

x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.shape)

print(torch.mv(A, x))
```

输出结果：

```shell
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.],
        [16., 17., 18., 19.]])
torch.Size([5, 4])

tensor([0., 1., 2., 3.])
torch.Size([4])

tensor([ 14.,  38.,  62.,  86., 110.])
```

### L2 范数

L2 范数是向量元素平方和的平方根

```python
import torch

u = torch.tensor([3.0, -4.0])

print(torch.norm(u))
```

输出结果：

```shell
tensor(5.)
```
