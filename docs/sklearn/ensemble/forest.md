# 随机森林 API

上一节我们提到了 Bagging 与决策树的结合随机森林，这是一种非常经典的集成学习方法。

由于过于经典，sklearn 内置了随机森林的实现，我们可以直接使用 sklearn 的 API 来使用随机森林。

## API

```python
sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None, bootstrap=True, random_state=None, min_samples_split=2)
```

其中：

- `n_estimators`：树的数量，即随机森林的棵树的数量。
- `criterion`：划分节点的标准，可以选择 `gini` 或 `entropy`。
- `max_depth`：树的最大深度。
- `bootstrap`：是否使用 bootstrap 采样。
- `random_state`：随机数种子。
- `min_samples_split`：节点分裂所需的最小样本数。

常用属性：

- `feature_importances_`：特征重要性。
- `estimators_`：随机森林中每棵树的模型。
- `n_classes_`：分类任务的类别数。
- `n_features_`：特征数。
- `n_outputs_`：输出的维度。

常用方法：

- `fit(X, y)`：训练随机森林模型。
- `predict(X)`：预测样本的类别。
- `predict_proba(X)`：预测样本的概率。
- `score(X, y)`：计算模型在测试集上的准确率。
