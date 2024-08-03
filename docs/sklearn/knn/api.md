# k 近邻算法 api

## API

```python
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
```

其中：

- `n_neighbors`：k 值，即取样本中最邻近的 k 个点。
- `weights`：权重方式，可选 `uniform` 或 `distance`，默认为 `uniform`。
- `algorithm`：算法类型，可选 `auto`、`ball_tree`、`kd_tree`、`brute`，默认为 `auto`。
- `leaf_size`：树的叶子节点的大小，默认为 30。
- `p`：距离度量的指数，默认为 2。
- `metric`：距离度量，可选 `minkowski`、`euclidean`、`manhattan`、`chebyshev`、`hamming`、`canberra`、`braycurtis`、`mahalanobis`、`wminkowski`、`seuclidean`、`cosine`、`correlation`、`haversine`、`hamming`、`jaccard`、`dice`、`russellrao`、`kulsinski`、`rogerstanimoto`、`sokalmichener`、`sokalsneath`、`yule`、`matching`、`jaccard`、`dice`、`kulsinski`、`rogerstanimoto`、`russellrao`、`sokalmichener`、`sokalsneath`、`yule`，默认为 `minkowski`。
- `metric_params`：距离度量参数，默认为 `None`。
- `n_jobs`：并行数，默认为 `None`。

常用方法：

- `fit(X, y)`：训练模型。
- `predict(X)`：预测标签。
- `predict_proba(X)`：预测概率。
- `kneighbors(X, n_neighbors=None, return_distance=True)`：返回 k 个最近邻的索引和距离。
- `radius_neighbors(X, radius=None, return_distance=True)`：返回半径内的点的索引和距离。
- `kneighbors_graph(X, n_neighbors=None, mode='connectivity', include_self=False)`：返回 k 近邻图。
- `radius_neighbors_graph(X, radius=None, mode='connectivity', include_self=False)`：返回半径内的点的 k 近邻图。

## 案例

```python
# 导入数据集
from sklearn.neighbors import KNeighborsClassifier

# 1.准备数据
x = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

# 2.实例化模型
# 实例化API
estimator = KNeighborsClassifier(n_neighbors=2)
# 使用fit方法进行训练
estimator.fit(x, y)

# 3.预测标签
print(estimator.predict([[1]]))
```

某一次的输出结果：

```shell
[0]
```
