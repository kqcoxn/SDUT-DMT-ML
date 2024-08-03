# 逻辑回归 api 介绍

## API

```python
sklearn.linear_model.LogisticRegression(solver='liblinear', penalty=‘l2’, C = 1.0)
```

其中：

- `solver`：指定优化算法，可选值为 `‘liblinear’` 或 `‘lbfgs’`，默认为 `‘liblinear’`。
- `penalty`：指定正则化项，可选值为 `‘l1’` 或 `‘l2’`，默认为 `‘l2’`。
- `C`：正则化系数，默认为 `1.0`。

默认将类别数量少的当做正例，类别数量多的当做反例。

LogisticRegression 方法相当于 SGDClassifier(loss="log", penalty=" ")，SGDClassifier 实现了一个普通的随机梯度下降学习。而使用 LogisticRegression(实现了 SAG)

使用方法与线性回归类似：

```python
# 导入模型
from sklearn.linear_model import LogisticRegression
# 实例化模型
model = LogisticRegression()
# 训练模型
model.fit(X_train, y_train)
# 预测
y_pred = model.predict(X_test)
```
