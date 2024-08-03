# 决策树算法 API

看起来我们定义了一堆公式，不过从工程的角度上看我们只需要直接调用一个简单的函数即可。

## API

```python
sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, presort=False)
```

其中：

- `criterion`：划分节点的评价标准，可以选择 `gini` 或 `entropy` 两种。
- `splitter`：选择最优切分点的方法，可以选择 `best` 或 `random` 两种。
- `max_depth`：树的最大深度，如果为 `None` 则表示树的深度不限制。
- `min_samples_split`：内部节点再划分所需的最小样本数。
- `min_samples_leaf`：叶子节点最少包含的样本数。
- `min_weight_fraction_leaf`：叶子节点最少包含的样本权重和。
- `max_features`：选择分裂时考虑的特征数，可以是整数或浮点数，也可以是 `auto`、`sqrt`、`log2` 或 `None`。
- `random_state`：随机数种子。
- `max_leaf_nodes`：最大叶子节点数。
- `min_impurity_decrease`：节点的纯度下降阈值。
- `class_weight`：样本权重，可以是字典或 `balanced`。
- `presort`：是否先对数据进行排序，以加快分裂速度。
