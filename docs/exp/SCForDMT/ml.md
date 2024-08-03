# 机器学习库基础

## 相关标准库和扩展库

用于数据分析、科学计算与可视化的扩展模块主要有：numpy、scipy、pandas、SymPy、matplotlib、Traits、TraitsUI、Chaco、TVTK、Mayavi、VPython、OpenCV。

- **numpy**：科学计算包，支持 N 维数组运算、处理大型矩阵、成熟的广播函数库、矢量运算、线性代数、傅里叶变换、随机数生成，并可与 C++/Fortran 语言无缝结合。树莓派 Python v3 默认安装已经包含了 numpy。
- **scipy**：scipy 依赖于 numpy，提供了更多的数学工具，包括矩阵运算、线性方程组求解、积分、优化、插值、信号处理、图像处理、统计等等。
- **matplotlib** 模块依赖于 numpy 模块和 tkinter 模块，可以绘制多种形式的图形，包括线图、直方图、饼状图、散点图、误差线图等等，图形质量可满足出版要求，是数据可视化的重要工具。
- **pandas**（Python Data Analysis Library）是基于 numpy 的数据分析模块，提供了大量标准数据模型和高效操作大型数据集所需要的工具，可以说 pandas 是使得 Python 能够成为高效且强大的数据分析环境的重要因素之一。

## Numpy 简单应用

### 生成数组

```python
import numpy as np

# 创建数组
a = np.array([1, 2, 3])
print(a)  # [1 2 3]

# 创建 2 维数组
b = np.array([[1, 2], [3, 4]])
print(b)  # [[1 2]
            #  [3 4]]


# 创建指定范围的数组
d = np.arange(10)
print(d)  # [0 1 2 3 4 5 6 7 8 9]

# 创建指定范围的数组
e = np.linspace(0, 1, 5)
print(e)  # [0.   0.25 0.5  0.75 1.  ]

# 创建随机数组
f = np.random.rand(3, 4)
print(f)  # [[0.57142857 0.71428571 0.64285714 0.54285714]
            #  [0.14285714 0.35714286 0.92857143 0.85714286]
            #  [0.21428571 0.14285714 0.78571429 0.64285714]]

# 创建单位矩阵
g = np.eye(3)
print(g)  # [[1. 0. 0.]
            #  [0. 1. 0.]
            #  [0. 0. 1.]]
```

### 测试两个数组是否足够接近

```python
import numpy as np

# 创建数组
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 测试两个数组是否足够接近
print(np.allclose(a, b))  # False

# 测试两个数组是否足够接近，允许误差为 1e-05
print(np.allclose(a, b, rtol=1e-05, atol=1e-08))  # True
```

### 改变数组元素值

```python
import numpy as np

# 创建数组
a = np.array([1, 2, 3])

# 修改数组元素值
a[0] = 5
print(a)  # [5 2 3]

b = np.array([[1, 2], [3, 4]])
b[0, 0] = 5
print(b)  # [[5 2]
            #  [3 4]]

# 利用条件语句修改数组元素值
a[a > 2] = 0
print(a)  # [5 0 0]


# 利用条件语句修改数组元素值
b[b > 2] = 0
print(b)  # [[5 0]
            #  [0 0]]
```

### 数组运算

```python
import numpy as np

# 创建数组
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 数组加法
c = a + b
print(c)  # [5 7 9]

# 数组减法
c = a - b
print(c)  # [-3 -3 -3]


# 数组乘法
c = a * b
print(c)  # [ 4 10 18]

# 数组除法
c = a / b
print(c)  # [0.25 0.4  0.5 ]


# 数组的幂运算
c = a ** 2
print(c)  # [1 4 9]

# 数组的矩阵乘法
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)
print(C)  # [[19 22]
            #  [43 50]]
```

### 常用方法

```python
import numpy as np

# 创建数组
a = np.array([1, 2, 3])

# 数组求和
print(np.sum(a))  # 6

# 数组求积
print(np.prod(a))  # 6

# 数组求平均值
print(np.mean(a))  # 2.0

# 数组求方差
print(np.var(a))  # 1.0

# 数组求标准差
print(np.std(a))  # 1.0

# 数组求最大值
print(np.max(a))  # 3

# 数组求最小值
print(np.min(a))  # 1

# 数组求排序后的数组
print(np.sort(a))  # [1 2 3]

# 数组转置
print(a.T)  # [1 2 3]

# 创建 2 维数组
b = np.array([[1, 2], [3, 4]])

# 2 维数组的行数
print(b.shape[0])  # 2

# 2 维数组的列数
print(b.shape[1])  # 2

# 2 维数组的转置
print(b.T)  # [[1 3]
            #  [2 4]]

# 2 维数组的维度
print(b.ndim)  # 2
```

### 函数运算

```python
import numpy as np

# 创建数组
a = np.array([1, 2, 3])

# 数组的正弦值
print(np.sin(a))  # [0.84147098 0.90929743 0.14112001]

# 数组的余弦值
print(np.cos(a))  # [0.54030231 0.31622777 0.9899925 ]

# 数组的正切值
print(np.tan(a))  # [1.55740772 -2.18503986  0.14254654]

# 数组的对数值
print(np.log(a))  # [0.        0.69314718 1.09861229]

# 数组的指数值
print(np.exp(a))  # [ 2.71828183  7.3890561  20.08553692]

# 数组的平方根
print(np.sqrt(a))  # [1. 1.41421356 1.73205081]

# 数组的绝对值
print(np.abs(a))  # [1 2 3]

# 数组的正弦值
print(np.arcsin(a))  # [ 1.57079633  1.04719755  0.32175055]
```

### 改变数组大小

```python
import numpy as np

a = np.array([1, 2, 3])

# 数组的维度
print(a.shape)  # (3,)

# 改变数组的维度
a.shape = (1, 3)
print(a)  # [[1 2 3]]


b = np.array([[1, 2], [3, 4]])

# 数组的维度
print(b.shape)  # (2, 2)


# 改变数组的维度
b.shape = (4, 1)
print(b)  # [[1]
            #  [2]
            #  [3]
            #  [4]]
```

### 切片

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a[1:5])  # [2 3 4 5]

b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(b[1:3, 1:3])  # [[5 6]
                    #  [8 9]]
print(b[1:3, :])  # [[4 5 6]
                  #  [7 8 9]]
print(b[:, 1:3])  # [[2 3]
                  #  [5 6]
                  #  [8 9]]
```

### 广播

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 数组加法
c = a + b
print(c)  # [5 7 9]

# 数组减法
c = a - b
print(c)  # [-3 -3 -3]

# 数组乘法
c = a * b
print(c)  # [ 4 10 18]

# 数组除法
c = a / b
print(c)  # [0.25 0.4  0.5 ]
```

### 计算唯一值以及出现次数

```python
import numpy as np

a = np.array([1, 2, 3, 2, 1, 4, 5, 4, 4, 4])

# 计算唯一值
print(np.unique(a))  # [1 2 3 4 5]

# 计算出现次数
print(np.bincount(a))  # [1 3 1 4 1]
```

### 矩阵运算

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵乘法
C = np.dot(A, B)
print(C)  # [[19 22]
            #  [43 50]]

# 矩阵的逆
C = np.linalg.inv(A)
print(C)  # [[-2.   1. ]
            #  [ 1.5 -0.5]]

# 矩阵的特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)  # [5. 9.]
print(eigenvectors)  # [[-0.70710678 -0.70710678]
                      #  [ 0.70710678  0.70710678]]
```

### 矩阵 QR 分解

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])

# 矩阵 QR 分解
Q, R = np.linalg.qr(A)
print(Q)  # [[-0.70710678 -0.70710678]
            #  [ 0.70710678  0.70710678]]
print(R)  # [[-3. -2.]
            #  [ 0.  1.]]

# 矩阵的逆
A_inv = np.linalg.inv(A)

# 验证矩阵乘法的正确性
A_inv_Q = np.dot(A_inv, Q)
print(A_inv_Q)  # [[ 1.  0.]
                 #  [ 0.  1.]]

# 验证矩阵乘法的正确性
Q_R = np.dot(Q, R)
print(Q_R)  # [[-3. -2.]
             #  [ 0.  1.]]
```

### 矩阵不同维度上的计算

1. 矩阵乘法：当两个矩阵的维度不同时，需要进行广播，使得两个矩阵的维度相同。
2. 矩阵的逆：当矩阵的维度大于 2 时，无法求逆，需要使用 SVD 分解或其他方法求逆。
3. 矩阵的特征值和特征向量：当矩阵的维度大于 2 时，无法求特征值和特征向量，需要使用 SVD 分解或其他方法求解。
4. 矩阵 QR 分解：当矩阵的维度大于 2 时，无法进行 QR 分解，需要使用 SVD 分解或其他方法求解。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([5, 6])

# 矩阵乘法
C = np.dot(A, B)
print(C)  # [17 39]

# 矩阵的逆
C = np.linalg.inv(A)
print(C)  # [[-2.   1. ]
            #  [ 1.5 -0.5]]

# 矩阵的特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)  # [5. 9.]
print(eigenvectors)  # [[-0.70710678 -0.70710678]
                      #  [ 0.70710678  0.70710678]]

# 矩阵 QR 分解
Q, R = np.linalg.qr(A)
print(Q)  # [[-0.70710678 -0.70710678]
            #  [ 0.70710678  0.70710678]]
print(R)  # [[-3. -2.]
            #  [ 0.  1.]]
```

### 常用常量

- `np.pi`：圆周率
- `np.inf`：正无穷大
- `np.nan`：非数值（Not a Number）
- `np.e`：自然常数
- `np.NINF`：负无穷大
- `np.PZERO`：正零
- `np.NZERO`：负零

## scipy 简单应用

scipy 主要模块：

| 模块        | 说明                                                                                                                                                |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| constants   | 常数                                                                                                                                                |
| special     | 特殊函数                                                                                                                                            |
| optimize    | 数值优化算法，如最小二乘拟合（leastsq）、函数最小值（fmin 系列）、非线性方程组求解（fsolve）等等                                                    |
| interpolate | 插值（interp1d、interp2d 等等）                                                                                                                     |
| integrate   | 数值积分                                                                                                                                            |
| signal      | 信号处理                                                                                                                                            |
| ndimage     | 图像处理，包括滤波器模块 filters、傅里叶变换模块 fourier、图像插值模块 interpolation、图像测量模块 measurements、形态学图像处理模块 morphology 等等 |
| stats       | 统计                                                                                                                                                |
| misc        | 提供读取图像文件的方法和一些测试图像                                                                                                                |
| io          | 提供读取 Matlab 和 Fortran 文件的方法                                                                                                               |

### 科学计算的常数

- `scipy.constants.e`：自然常数
- `scipy.constants.m_e`：电子质量
- `scipy.constants.c`：光速
- `scipy.constants.h`：普朗克常数
- `scipy.constants.k`：玻尔兹曼常数
- `scipy.constants.G`：万有引力常数
- `scipy.constants.pi`：圆周率

### 常用函数

- `scipy.special.cbrt(x)`：计算 x 的立方根
- `scipy.special.exp1(x)`：计算 e 的 x 次幂
- `scipy.special.expn(n, x)`：计算 e 的 n 次幂
- `scipy.special.gammaln(x)`：计算 x 的伽玛函数的自然对数
- `scipy.special.loggamma(x)`：计算 x 的伽玛函数的对数
- `scipy.special.erf(x)`：计算误差函数
- `scipy.special.erfc(x)`：计算补充误差函数
- `scipy.special.erfinv(x)`：计算 x 的反正切值

## 数据分析模块 pandas

pandas 主要提供了 3 种数据结构：1）Series，带标签的一维数组；2）DataFrame，带标签且大小可变的二维表格结构；3）Panel，带标签且大小可变的三维数组。

### 生成一维数组

```python
import pandas as pd

# 生成一维数组
s = pd.Series([1, 2, 3, 4, 5])
print(s)  # 0    1
            # 1    2
            # 2    3
            # 3    4
            # 4    5
            # dtype: int64
```

### 生成 DataFrame

```python
import pandas as pd

# 生成 DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df)  #    A  B
         # 0  1  4
         # 1  2  5
         # 2  3  6
```

### 二维数据查看

```python
import pandas as pd

# 生成 DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# 列名
print(df.columns)  # Index(['A', 'B'], dtype='object')

# 行名
print(df.index)  # RangeIndex(start=0, stop=3, step=1)

# 值
print(df.values)  # [[1 4]
                 #  [2 5]
                 #  [3 6]]

# 行数
print(len(df))  # 3

# 列数
print(df.shape[1])  # 2
```

### 查看二维数据的索引、列名和数据

```python
import pandas as pd

# 生成 DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# 列名
print(df.columns)  # Index(['A', 'B'], dtype='object')

# 行名
print(df.index)  # RangeIndex(start=0, stop=3, step=1)

# 值
print(df.values)  # [[1 4]
                 #  [2 5]
                 #  [3 6]]

# 行数
print(len(df))  # 3

# 列数
print(df.shape[1])  # 2
```

### 查看数据的统计信息

```python
import pandas as pd

# 生成 DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# 统计信息
print(df.describe())
```

### 二维数据转置

```python
import pandas as pd

# 生成 DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# 转置
df_T = df.T
print(df_T)
```

### 排序

```python
import pandas as pd

# 生成 DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# 排序
df_sorted = df.sort_values(by='A')
print(df_sorted)
```

### 选择数据

```python
import pandas as pd

# 生成 DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# 选择数据
df_A = df['A']
print(df_A)  # 0    1
            # 1    2
            # 2    3
            # Name: A, dtype: int64

# 选择数据
df_AB = df[['A', 'B']]
print(df_AB)
```

### 数据修改

```python
import pandas as pd

# 生成 DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# 修改数据
df['A'] = [10, 20, 30]
print(df)  #    A  B
    0  10  4
    1  20  5
    2  30  6

# 修改数据
df.loc[0, 'A'] = 100
print(df)
```

### 缺失值处理

```python
import pandas as pd

# 生成 DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

# 缺失值处理
df_dropna = df.dropna()
print(df_dropna)


# 缺失值处理
df_fillna = df.fillna(value=0)
print(df_fillna)
```

### 其他常用操作

- `df.head(n)`：查看前 n 行数据
- `df.tail(n)`：查看后 n 行数据
- `df.sample(n)`：随机抽样 n 行数据
- `df.groupby(by)`：按 by 列进行分组
- `df.merge(other, on)`：合并两个 DataFrame
- `df.pivot_table(index, columns, values)`：生成透视表
- `df.plot(kind)`：绘制图表
- `df.cut(data, category, right=False)`：数据离散化

## matplotlib 简单应用

matplotlib 模块依赖于 numpy 模块和 tkinter 模块，可以绘制多种形式的图形，包括线图、直方图、饼状图、散点图、误差线图等等。

matplotlib 库提供的图形非常多，用到相关功能的时候再去查阅文档即可。

### 常用 API

- `matplotlib.pyplot.plot(x, y)`：绘制线图
- `matplotlib.pyplot.hist(x, bins=10)`：绘制直方图
- `matplotlib.pyplot.bar(x, y)`：绘制条形图
- `matplotlib.pyplot.scatter(x, y)`：绘制散点图
- `matplotlib.pyplot.errorbar(x, y, yerr)`：绘制误差线图
- `matplotlib.pyplot.pie(x, explode)`：绘制饼状图
- `matplotlib.pyplot.imshow(x)`：绘制图像
- `matplotlib.pyplot.show()`：显示图形

例如：

```python
import matplotlib.pyplot as plt

# 生成数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 绘制线图
plt.plot(x, y)
plt.show()

# 绘制直方图
plt.hist(y, bins=5)
plt.show()
```

## 补充 1：使用线性回归拟合平面最佳直线及预测

代码采用 sklearn 扩展库实现，使用线性回归算法解决下面的问题：根据平面上已知 3 个点的坐标，拟合最佳直线斜率 k 和截距 b，然后根据拟合的结果对给出的 x 坐标进行预测，得到 y 坐标。

```python
from sklearn import linear_model

def linearRegressionPredict(x, y):
    lr = linear_model.LinearRegression()
    # 拟合
    lr.fit(x, y)
    return lr

# 平面上三个点的x轴坐标
x = [[1], [5], [7]]
# 平面上三个点的y轴坐标
y = [[3], [100], [120]]

# 根据已知3个点拟合最佳直线的系数和截距
lr = linearRegressionPredict(x, y)
# 查看最佳拟合系数
print('k:', lr.coef_)
# 截距
print('b:', lr.intercept_)

# 测试代码，预测
xs = [[[3]], [[5]], [[7]], [[10]]]
for item in xs:
    print(item, ':', lr.predict(item))
```

某一次的运行结果：

```shell
运行结果：
k: [[ 20.17857143]]
b: [-13.10714286]
[[3]] : [[ 47.42857143]]
[[5]] : [[ 87.78571429]]
[[7]] : [[ 128.14285714]]
[[10]] : [[ 188.67857143]]
```

## 补充 2：Python+sklearn 使用线性回归算法预测儿童身高

问题描述：一个人的身高除了随年龄变大而增长之外，在一定程度上还受到遗传和饮食以及其他因素的影响，代码中假定受年龄、性别、父母身高、祖父母身高和外祖父母身高共同影响，并假定大致符合线性关系。

```python
import copy
import numpy as np
from sklearn import linear_model

def linearRegressionPredict(x, y):
    lr = linear_model.LinearRegression()
    # 拟合
    lr.fit(x, y)
    return lr
# 儿童年龄,性别（0女1男）,父亲身高,母亲身高,祖父身高,祖母身高,外祖父身高,外祖母身高
x = np.array([[1, 0, 180, 165, 175, 165, 170, 165],\
              [3, 0, 180, 165, 175, 165, 173, 165],\
              [4, 0, 180, 165, 175, 165, 170, 165],\
              [6, 0, 180, 165, 175, 165, 170, 165],\
              [8, 1, 180, 165, 175, 167, 170, 165],\
              [10, 0, 180, 166, 175, 165, 170, 165],\
              [11, 0, 180, 165, 175, 165, 170, 165],\
              [12, 0, 180, 165, 175, 165, 170, 165],\
              [13, 1, 180, 165, 175, 165, 170, 165],\
              [14, 0, 180, 165, 175, 165, 170, 165],\
              [17, 0, 170, 165, 175, 165, 170, 165]])
# 儿童身高，单位：cm
y = np.array([60, 90, 100, 110,\
              130, 140, 150, 164,\
              160, 163, 168])
# 根据已知数据拟合最佳直线的系数和截距
lr = linearRegressionPredict(x, y)
# 查看最佳拟合系数
print('k:', lr.coef_)
# 截距
print('b:', lr.intercept_)
# 预测
xs = np.array([[10, 0, 180, 165, 175, 165, 170, 165],\
               [17, 1, 173, 153, 175, 161, 170, 161],\
               [34, 0, 170, 165, 170, 165, 170, 165]])
for item in xs:
    # 深复制，假设超过18岁以后就不再长高了
    item1 = copy.deepcopy(item)
    if item1[0] > 18:
        item1[0] = 18
    print(item, ':', lr.predict(item1.reshape(1,-1)))
```

某一次的运行结果：

```shell
k: [  8.03076923e+00  -4.65384615e+00   2.87769231e+00  -5.61538462e-01
   7.10542736e-15   5.07692308e+00   1.88461538e+00   0.00000000e+00]
b: -1523.15384615
[ 10   0 180 165 175 165 170 165] : [ 140.56153846]
[ 17   1 173 153 175 161 170 161] : [ 158.41]
[ 34   0 170 165 170 165 170 165] : [ 176.03076923]
```

## 补充 3：KNN 分类算法实现根据身高和体重对体型分类

KNN 算法是 k-Nearest Neighbor Classification 的简称，也就是 k 近邻分类算法。基本思路是在特征空间中查找 k 个最相似或者距离最近的样本，然后根据 k 个最相似的样本对未知样本进行分类。基本步骤为：

1. 计算已知样本空间中所有点与未知样本的距离；
2. 对所有距离按升序排列；
3. 确定并选取与未知样本距离最小的 k 个样本或点；
4. 统计选取的 k 个点所属类别的出现频率；
5. 把出现频率最高的类别作为预测结果，即未知样本所属类别。

下面的代码模拟了上面的算法思路和步骤，以身高+体重对肥胖程度进行分类为例，采用欧几里得距离。

```python
# 使用sklearn库的k近邻分类模型
from sklearn.neighbors import KNeighborsClassifier

# 创建并训练模型
clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
clf.fit(knownData, knownTarget)

# 分类
for current in unKnownData:
    print(current, end=' : ')
    current = np.array(current).reshape(1,-1)
    print(clf.predict(current)[0])
```

某一次的运行结果：

```shell
(1, 180, 70) : 偏瘦
(1, 160, 90) : 过胖
(1, 170, 85) : 正常
```

## 补充 4：绘制时间序列数据的时序图、自相关图和偏自相关图

时序图、自相关图和偏相关图是判断时间序列数据是否平稳的重要依据。

```python
def generateData(startDate, endDate):
    df = pd.DataFrame([300+i*30+randrange(50) for i in range(31)],\
                      columns=['营业额'],\
                      index=pd.date_range(startDate, endDate, freq='D'))
    return df

# 生成测试数据，模拟某商店营业额
data = generateData('20170601',  '20170701')
print(data)

# 绘制时序图
myfont = fm.FontProperties(fname=r'C:\Windows\Fonts\STKAITI.ttf')
data.plot()
plt.legend(prop=myfont) 
plt.show()
# 绘制自相关图
plot_acf(data).show()
# 绘制偏自相关图
plot_pacf(data).show()
```

## 补充 5：使用系统聚类算法对随机元素进行分类

- 系统聚类算法又称层次聚类或系谱聚类，首先把样本看作各自一类，定义类间距离，选择距离最小的一对元素合并成一个新的类，重复计算各类之间的距离并重复上面的步骤，直到将所有原始元素分成指定数量的类。
- 该算法的计算复杂度比较高，不适合大数据聚类问题。

```python
def generateData():
    '''生成测试数据'''
    def get(start, end):
        return [randrange(start, end) for _ in range(30)]

    x1 = get(0, 40)
    x2 = get(70, 100)
    y1 = get(0, 30)
    y2 = get(40, 70)

    data = list(zip(x1, y1)) + list(zip(x1, y2))+\
           list(zip(x2, y1)) + list(zip(x2, y2))
    return np.array(data)

def AgglomerativeTest(n_clusters):
    '''聚类，指定类的数量，并绘制图形'''
    assert 1 <= n_clusters <= 4
    predictResult = AgglomerativeClustering(n_clusters=n_clusters,
                                            affinity='euclidean',
                                            linkage='ward').fit_predict(data)
    colors = 'rgby'
    markers = 'o*v+'
    for i in range(n_clusters):
        subData = data[predictResult==i]
        plt.scatter(subData[:,0], subData[:,1], c=colors[i], marker=markers[i], s=40)
    plt.show()

# 生成随机数据
data = generateData()
# 聚类为3个不同的类
AgglomerativeTest(3)
# 聚类为4个不同的类
AgglomerativeTest(4)
```

## 补充 6：使用 k-means 聚类算法进行分类

K-means 算法的基本思想是：以空间中 k 个点为中心进行聚类，对最靠近他们的对象归类。通过迭代的方法，逐次更新各聚类中心的值，直至得到最好的聚类结果。

最终的 k 个聚类具有以下特点：各聚类本身尽可能的紧凑，而各聚类之间尽可能的分开。

该算法的最大优势在于简洁和快速，算法的关键在于预期分类数量的确定以及初始中心和距离公式的选择。

假设要把样本集分为 c 个类别，算法描述如下：

1. 适当选择 c 个类的初始中心；
2. 在第 k 次迭代中，对任意一个样本，求其到 c 个中心的距离，将该样本归到距离最短的中心所在的类；
3. 利用均值等方法更新该类的中心值；
4. 对于所有的 c 个聚类中心，如果利用（2）（3）的迭代法更新后，值保持不变，则迭代结束，否则继续迭代。

```python
from numpy import array
from random import randrange
from sklearn.cluster import KMeans

# 获取模拟数据
X = array([[1,1,1,1,1,1,1],
           [2,3,2,2,2,2,2],
           [3,2,3,3,3,3,3],
           [1,2,1,2,2,1,2],
           [2,1,3,3,3,2,1],
           [6,2,30,3,33,2,71]])

# 训练
kmeansPredicter = KMeans(n_clusters=3).fit(X)
# 原始数据分类
category = kmeansPredicter.predict(X)
print('分类情况：', category)
print('='*30)

def predict(element):
    result = kmeansPredicter.predict(element)
    print('预测结果：', result)
    print('相似元素：\n', X[category==result])

# 测试
predict([[1,2,3,3,1,3,1]])
print('='*30)
predict([[5,2,23,2,21,5,51]])
```
