# 线性回归 API 再介绍

经过前两节的介绍，我们发现线性回归似乎并不想我们想的那样简单，就算是调用 API 我们也有很多参数需要细微的调节。

## 线性回归模型

### 正规方程法

```python
sklearn.linear_model.LinearRegression(fit_intercept=True)
```

其中：

- `fit_intercept`：是否包含截距项。默认值为 `True`

关键属性：

- `coef_`：回归系数
- `intercept_`：偏置

### 梯度下降法

```python
sklearn.linear_model.SGDRegressor(loss="squared_loss", fit_intercept=True, learning_rate ='invscaling', eta0=0.01)
```

SGDRegressor 类实现了随机梯度下降学习，它支持不同的 loss 函数和正则化惩罚项来拟合线性回归模型。

其中：

- `loss`：损失函数，默认为 `squared_loss`，即平方损失函数。
- `fit_intercept`：是否包含截距项。默认值为 `True`
- `learning_rate`：学习率的策略，默认为 `invscaling`，即随着迭代次数的增加，学习率逐渐减小。
- `eta0`：初始学习率。

关键属性：

- `coef_`：回归系数
- `intercept_`：偏置

例如：

```python
from sklearn.linear_model import SGDRegressor

# 模拟数据
X = [[1, 2], [3, 4], [5, 6]]
y = [1, 2, 3]

# 实例化SGDRegressor
regressor = SGDRegressor()

# 训练模型
regressor.fit(X, y)

# 预测结果
y_pred = regressor.predict([[7, 8]])
print(y_pred)

# 属性
print(regressor.coef_)
print(regressor.intercept_)
```

输出结果：

```shell
[4.09384452]
[0.23028114 0.30135073]
[0.07107068]
```

## 辅助方法

在实际使用时，我们需要对模型进行一些预处理，比如标准化、归一化等，这时我们可以用到 sklearn 为我们提供的一些辅助方法。

我们将在下一节的例子中使用到他们。

### 均方误差(Mean Squared Error, MSE)评价

```python
sklearn.metrics.mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
```

其中：

- `y_true`：真实值
- `y_pred`：预测值
- `sample_weight`：样本权重，默认为 `None`
- `multioutput`：指定输出的计算方式，默认为 `'uniform_average'`，即均值。

### 数据集划分

```python
sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=None, shuffle=True, stratify=None)
```

其中：

- `X`：特征数据
- `y`：标签数据
- `test_size`：测试集占比，默认为 `0.25`
- `random_state`：随机种子，默认为 `None`
- `shuffle`：是否打乱数据，默认为 `True`
- `stratify`：是否分层抽样，默认为 `None`

返回训练集与测试集

### 标准化

```python
sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
```

其中：

- `copy`：是否复制数据，默认为 `True`
- `with_mean`：是否中心化，默认为 `True`
- `with_std`：是否标准化，默认为 `True`
