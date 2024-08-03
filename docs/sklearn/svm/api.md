# SVM API

## API

```python
sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
```

其中：

- `C`：软间隔参数，控制正则化强度，默认值为 1.0。
- `kernel`：核函数类型，默认值为'rbf'，可选'linear', 'poly', 'rbf','sigmoid'。
- `degree`：多项式核函数的最高次项，默认值为 3。
- `gamma`：核函数的系数，默认值为'scale'，也可指定具体的值。
- `shrinking`：是否使用收缩算法，默认值为 True。
- `probability`：是否返回预测的概率，默认值为 False。
- `tol`：容忍度，默认值为 0.001。
- `cache_size`：缓存大小，默认值为 200。
- `class_weight`：类别权重，默认值为 None。
- `verbose`：是否显示训练过程，默认值为 False。
- `max_iter`：最大迭代次数，默认值为-1。
- `decision_function_shape`：决策函数的形状，默认值为'ovr'，也可指定'ovo'。
- `random_state`：随机种子，默认值为 None。

## 示例

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建SVM分类器
clf = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)
print(y_pred)

# 评估模型
print(accuracy_score(y_test, y_pred))
```

某一次的输出结果：

```shell
[1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]
1.0
```
