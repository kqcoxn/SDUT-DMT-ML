# 特征工程-特征提取

决策树的数据必须是数值型的，但很多时候我们的数据是文本或图像等非数值型数据，这时我们需要对数据进行特征提取，将非数值型数据转换为数值型数据。

## 特征提取

将任意数据（如文本或图像）转换为可用于机器学习的数字特征，特征值化是为了计算机更好的去理解数据

特征提取分类:

- 字典特征提取(特征离散化)
- 文本特征提取
- 图像特征提取（深度学习将介绍）

sklearn 提供了非常方便的特征提取 API `sklearn.feature_extraction`

## 字典特征提取

作用：对字典数据进行特征值化

### API

```python
sklearn.feature_extraction.DictVectorizer(sparse=True, dtype=<class 'numpy.float64'>)
```

其中：

- `sparse`：是否返回稀疏矩阵，默认为 True
- `dtype`：数据类型，默认为 numpy.float64

常用方法：

1. `fit_transform(X)`：训练并转换数据
2. `transform(X)`：转换数据
3. `get_feature_names()`：获取特征名称

### 示例

我们对以下数据进行特征提取：

```python
from sklearn.feature_extraction import DictVectorizer

data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]

# 1、实例化一个转换器类
transfer = DictVectorizer(sparse=False)
# 2、调用fit_transform
data = transfer.fit_transform(data)

print("结果：\n", data)
print("特征名字：\n", transfer.feature_names_)
```

输出结果：

```shell
[[  0.   1.   0. 100.]
 [  1.   0.   0.  60.]
 [  0.   0.   1.  30.]]
特征名字：
 ['city=上海', 'city=北京', 'city=深圳', 'temperature']
```

之前在学习 pandas 中的离散化的时候，也实现了类似的效果。

我们把这个处理数据的技巧叫做”one-hot“编码：

![one-hot](../images/tree/onehot.png)

转化为：

![one-hot-encoding](../images/tree/onehot1.png)

对于特征当中存在类别信息的我们都会做 one-hot 编码处理

## 文本特征提取

作用：对文本数据进行特征值化

### API

```python
sklearn.feature_extraction.text.CountVectorizer(stop_words=[])
```

其中：

- `stop_words`：停用词列表，默认为空

常用方法：

1. `fit_transform(X)`：训练并转换数据
2. `transform(X)`：转换数据
3. `get_feature_names()`：获取特征名称

### 示例

我们对以下数据进行特征提取：

```python
from sklearn.feature_extraction.text import CountVectorizer

data = ["life is short,i like like python", "life is too long,i dislike python"]

# 1、实例化一个转换器类
# transfer = CountVectorizer(sparse=False)
transfer = CountVectorizer()
# 2、调用fit_transform
data = transfer.fit_transform(data)

print("文本特征抽取的结果：\n", data.toarray())
print("返回特征名字：\n", transfer.get_feature_names_out())
```

输出结果：

```shell
文本特征抽取的结果：
 [[0 1 1 2 0 1 1 0]
 [1 1 1 0 1 1 0 1]]
返回特征名字：
 ['dislike' 'is' 'life' 'like' 'long' 'python' 'short' 'too']
```

为什么会输出这样的结果呢？

因为默认的分词器是空格，所以会把每个词当做一个特征，然后统计每个词出现的次数。

## jieba 分词处理

在中文语境下，单词与汉字并不能等同，而应该与词组、短语等等进行分割。

jieba 是一款开源的中文分词工具，可以方便地对文本进行分词处理。

### API

安装 jieba：

```shell
pip install jieba
```

使用：

```python
jieba.cut(text, cut_all=False)
```

其中：

- `text`：待分词的文本
- `cut_all`：是否全模式分词，默认是精确模式

### 示例

对以下三句话进行特征值化：

> 今天很残酷，明天更残酷，后天很美好，
> 但绝对大部分是死在明天晚上，所以每个人不要放弃今天。
>
> 我们看到的从很远星系来的光是在几百万年之前发出的，
> 这样当我们看到宇宙时，我们是在看它的过去。
>
> 如果只用一种方式了解某样事物，你就不会真正了解它。
> 了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。

代码如下：

```python
from sklearn.feature_extraction.text import CountVectorizer
import jieba


def cut_word(text):
    # 用结巴对中文字符串进行分词
    text = " ".join(list(jieba.cut(text)))

    return text


data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但对大部分是死在明天晚上，所以每个人不要放弃今天。",
        "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
        "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

# 将原始数据转换成分好词的形式
text_list = []
for sent in data:
    text_list.append(cut_word(sent))
print(text_list)

# 1、实例化一个转换器类
# transfer = CountVectorizer(sparse=False)
transfer = CountVectorizer()
# 2、调用fit_transform
data = transfer.fit_transform(text_list)

print("文本特征抽取的结果：\n", data.toarray())
print("返回特征名字：\n", transfer.get_feature_names_out())
```

输出结果：

```shell
...notice...
['一种 还是 一种 今天 很 残酷 ， 明天 更 残酷 ， 后天 很 美好 ， 但 对 大部分 是 死 在 明天 晚上 ， 所以 每个 人 不要 放弃 今天 。', '我们 看到 的 从 很 远 星系 来 的 光是在 几百万年 之前 发出 的 ， 这样 当 我们 看到 宇宙 时 ， 我们 是 在 看 它 的 过去 。', '如果 只用 一种 方式 了解 某样 事 物 ， 你 就 不会 真正 了解 它 。 了解 事物 真正 含义 的 秘密 取决于 如何 将 其 与 我们 所 了解 的 事物 相 联系 。']
文本特征抽取的结果：
 [[2 0 1 0 0 0 2 0 0 0 0 0 1 0 1 0 0 0 0 1 1 0 2 0 1 0 2 1 0 0 0 1 0 0 1 0]
 [0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 1 3 0 0 0 0 1 0 0 0 0 2 0 0 0 0 1 0 1]
 [1 1 0 0 4 3 0 0 0 0 1 1 0 1 0 1 1 0 1 0 0 1 0 0 0 1 0 0 0 2 1 0 1 0 0 0]]
返回特征名字：
 ['一种' '不会' '不要' '之前' '了解' '事物' '今天' '光是在' '几百万年' ' 发出' '取决于' '只用' '后天' '含义'
 '大部分' '如何' '如果' '宇宙' '我们' '所以' '放弃' '方式' '明天' '星系' '晚上' '某样' '残酷' '每个'
 '看到' '真正' '秘密' '美好' '联系' '过去' '还是' '这样']
```

但如果把这样的词语特征用于分类，会出现什么问题？

![text-feature-problem](../images/tree/词语占比.png)

该如何处理某个词或短语在多篇文章中出现的次数高这种情况？

## Tf-idf 文本特征提取

TF-IDF 的主要思想是：如果某个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。

TF-IDF 作用：用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。

### 公式

![tf-idf-formula](../images/tree/tfidf公式.png)

- 词频（term frequency，tf）指的是某一个给定的词语在该文件中出现的频率
- 逆向文档频率（inverse document frequency，idf）是一个词语普遍重要性的度量。某一特定词语的 idf，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取以 10 为底的对数得到

最终得出结果可以理解为重要程度。

例如，假如一篇文章的总词语数是 100 个，而词语"非常"出现了 5 次，那么"非常"一词在该文件中的词频就是 5/100=0.05，而计算文件频率（IDF）的方法是以文件集的文件总数，除以出现"非常"一词的文件数。所以，如果"非常"一词在 1,0000 份文件出现过，而文件总数是 10,000,000 份的话，其逆向文件频率就是 lg（10,000,000 / 1,0000）=3，最后"非常"对于这篇文档的 tf-idf 的分数为 0.05 \* 3=0.15

### API

```python
sklearn.feature_extraction.text.TfidfVectorizer(stop_words=[], max_features=None, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
```

其中：

- `stop_words`：停用词列表，默认为空
- `max_features`：最大特征数，默认为 None
- `norm`：归一化方法，默认为 l2
- `use_idf`：是否使用 idf 信息，默认为 True
- `smooth_idf`：是否平滑 idf，默认为 True
- `sublinear_tf`：是否使用平滑的词频，默认为 False

常用方法：

1. `fit_transform(X)`：训练并转换数据
2. `transform(X)`：转换数据
3. `get_feature_names()`：获取特征名称

### 示例

还是刚才的例子：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba


def cut_word(text):
    text = " ".join(list(jieba.cut(text)))
    return text


data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但对大部分是死在明天晚上，所以每个人不要放弃今天。",
        "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
        "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

# 将原始数据转换成分好词的形式
text_list = []
for sent in data:
    text_list.append(cut_word(sent))
print(text_list)

# 1、实例化一个转换器类
# transfer = CountVectorizer(sparse=False)
transfer = TfidfVectorizer(stop_words=['一种', '不会', '要'])
# 2、调用fit_transform
data = transfer.fit_transform(text_list)

print("文本特征抽取的结果：\n", data.toarray())
print("返回特征名字：\n", transfer.get_feature_names_out())
```

输出结果：

```shell
...notice...
['一种 还是 一种 今天 很 残酷 ， 明天 更 残酷 ， 后天 很 美好 ， 但 对 大部分 是 死 在 明天 晚上 ， 所以 每个 人 不要 放弃 今天 。', '我们 看到 的 从 很 远 星系 来 的 光是在 几百万年 之前 发出 的 ， 这样 当 我们 看到 宇宙 时 ， 我们 是 在 看 它 的 过去 。', '如果 只用 一种 方式 了解 某样 事 物 ， 你 就 不会 真正 了解 它 。 了解 事物 真正 含义 的 秘密 取决于 如何 将 其 与 我们 所 了解 的 事物 相 联系 。']
文本特征抽取的结果：
 [[0.21821789 0.         0.         0.         0.43643578 0.
  0.         0.         0.         0.         0.21821789 0.
  0.21821789 0.         0.         0.         0.         0.21821789
  0.21821789 0.         0.43643578 0.         0.21821789 0.
  0.43643578 0.21821789 0.         0.         0.         0.21821789
  0.         0.         0.21821789 0.        ]
 [0.         0.2410822  0.         0.         0.         0.2410822
  0.2410822  0.2410822  0.         0.         0.         0.
  0.         0.         0.         0.2410822  0.55004769 0.
  0.         0.         0.         0.2410822  0.         0.
  0.         0.         0.48216441 0.         0.         0.
  0.         0.2410822  0.         0.2410822 ]
 [0.         0.         0.644003   0.48300225 0.         0.
  0.         0.         0.16100075 0.16100075 0.         0.16100075
  0.         0.16100075 0.16100075 0.         0.12244522 0.
  0.         0.16100075 0.         0.         0.         0.16100075
  0.         0.         0.         0.3220015  0.16100075 0.
  0.16100075 0.         0.         0.        ]]
返回特征名字：
 ['不要' '之前' '了解' '事物' '今天' '光是在' '几百万年' '发出' '取决于' '只用' '后天' '含义' '大部分'
 '如何' '如果' '宇宙' '我们' '所以' '放弃' '方式' '明天' '星系' '晚上' ' 某样' '残酷' '每个' '看到'
 '真正' '秘密' '美好' '联系' '过去' '还是' '这样']
```
