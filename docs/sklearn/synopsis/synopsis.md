# 从这里开始你的 ML 与 SciKit-Learn 学习之旅！

欢迎来到机器学习！

## 关于本文档

本文档旨在用开发者的视角学习机器学习与 sklearn 库的相关知识，相比于教材极大的缩减了原理推导部分，只保留三个核心问题，即“**为什么可以这样用**”、“**什么时候需要这样用**”与“**怎样正确的使用**”

总而言之，如果只把机器学习当作一个工具而不是科研方向，本文档可能更加合适。

## 什么是机器学习？

机器学习(Machine Learning) 是一门研究如何使计算机基于数据自动分析、学习和改进，并最终做出预测或决策的学科。 机器学习算法通常分为三类：

1. **监督学习**(Supervised Learning)：监督学习是指由训练数据（带有正确答案的输入-输出对）驱动的学习过程，目的是对输入数据进行正确的预测或分类。监督学习的算法包括线性回归、逻辑回归、决策树、神经网络、支持向量机、K-近邻、朴素贝叶斯等。
2. **无监督学习**(Unsupervised Learning)：无监督学习是指对数据进行分析而不进行任何先验假设的学习过程，目的是对数据进行聚类、降维、概率估计等。无监督学习的算法包括聚类、K-均值、高斯混合模型、DBSCAN、谱聚类等。
3. **强化学习**(Reinforcement Learning)：强化学习是指机器通过与环境的互动来学习，并在此过程中不断调整策略以最大化预期的回报。强化学习的算法包括 Q-learning、策略梯度、Actor-Critic 等。

机器学习的应用场景主要有：

- **计算机领域**：图像识别、文本分析、语音识别、推荐系统、生物信息学、金融、保险、医疗、互联网搜索、生物制药、智能助理等。
- **交互设计**：机器人、自动驾驶、智能助手、智能家居、智能电视、智能手机、智能穿戴等。
- **数据领域**：广告、推荐系统、搜索引擎、垃圾邮件过滤、图像识别、语音识别、文本分析、图像分类、语音识别、自然语言处理、情感分析等。

在本文档中，以上机器学习的相关概念和术语将会逐一介绍，并结合 sklearn 库进行实践。

## 什么是 SciKit-Learn？

Scikit-Learn(sklearn) 是基于 Python 语言的机器学习工具。它建立在 NumPy, SciPy, Pandas 和 Matplotlib 之上，包含分类、回归、聚类、降维、模型选择和预处理等算法 API，所有对象的接口简单，很适合新手上路。

所以，**sklearn 是机器学习的工具包，将一系列常用的机器学习算法进行了整合，我们可以通过简单调用 API 的方式实现复杂的机器学习算法**。

**官网地址**：[scikit-learn.org](https://scikit-learn.org)

**开源地址**：[scikit-learn](https://github.com/scikit-learn/scikit-learn)

## 为什么是 Scikit-Learn？

Scikit-Learn 是一个开源的 Python 库，它提供了许多机器学习算法，包括分类、回归、聚类、降维、模型选择和预处理等。

它有以下优点：

- **简单易用**：Scikit-Learn 的 API 设计简洁易懂，学习曲线平滑，文档齐全，适合新手学习。
- **功能丰富**：Scikit-Learn 提供了许多机器学习算法，包括分类、回归、聚类、降维、模型选择和预处理等，覆盖了机器学习领域的大部分内容。
- **开源免费**：Scikit-Learn 是一个开源项目，其代码遵循 BSD 协议，你可以免费使用和修改它。
- **社区活跃**：Scikit-Learn 是一个活跃的社区，其开发者和用户都在积极参与到项目的开发中。

如果你之前接触过机器学习，很大概率你也知道 tenserflow、PyTorch、Keras 等框架。这些库各有优势，但对于初学者来说，sklearn 是一个更好的选择，他包含了机器学习的绝大多数算法，更适合中小规模的数据集，而且文档、API 设计都很简单易懂，非常适合新手学习。

特别的，尽管 scikit-learn 是用 Python 编写的，但它的大部分核心算法是用 Cython 实现的，因此具有较高的性能。

## 学习 Scikit-Learn 有什么用？

问得好，说实话目前来看只对读研有用。

当然硬要说也是有的：

- **掌握基础和高级机器学习算法**：通过使用 scikit-learn 可以实践我们在课堂上学到的各种机器学习算法，从最基础的线性回归到高级的支持向量机和随机森林，有助于理解不同算法的原理和应用场景。

- **数据分析与处理能力**：scikit-learn 提供了丰富的数据预处理和特征工程工具，帮助你在实际项目中处理和分析数据，这种能力在科研和实际工作中非常重要。

- **项目实践经验**：通过 scikit-learn，你可以进行各种机器学习项目实践，如分类、回归、聚类、降维等。这些项目可以作为你的学习成果和工作中的实际案例，为你的简历增加亮点。

- **科研与论文写作**：在学术研究中，scikit-learn 可以帮助你快速实现和验证机器学习模型，提高科研效率，并为论文提供实验数据和分析结果。

- **提升编程技能**：scikit-learn 基于 Python 语言，学习它有助于提升你的 Python 编程技能。同时，它与 NumPy、SciPy、matplotlib 等库的结合使用，可以让你更加熟练地进行科学计算和数据可视化。

所以如果要读研必须要学，~~不读研学了也没太大用处。~~

当然最重要的，**它有 3 学分**，为了过课程至少得看的动代码是在干啥。

## Scikit-Learn 支持的功能

sklearn 的功能主要分为 6 大类：

1. **分类**：包括 Logistic 回归、支持向量机、K-近邻、朴素贝叶斯、决策树、随机森林、AdaBoost、GBDT、XGBoost 等。
2. **回归**：包括线性回归、岭回归、Lasso 回归、Ridge 回归、ElasticNet 回归、SGD 回归、ARD 回归、Bayesian 回归等。
3. **聚类**：包括 K-Means、层次聚类、DBSCAN、Mean Shift、Spectral Clustering 等。
4. **降维**：包括 PCA、SVD、ICA、t-SNE 等。
5. **模型选择**：包括交叉验证、网格搜索、超参数优化、度量指标、学习曲线等。
6. **预处理**：包括特征缩放、标准化、缺失值处理、特征提取、特征选择等。

其中，分类和回归是典型的**监督学习**算法，聚类、降维是典型的**无监督学习**算法。

在实际学习与应用时，我们并**不需要一次性全部学会，只需要在用到时查看[官方文档](https://scikit-learn.org/stable/api/index.html)** 即可。

## 开始学习！

上手实践永远是最好的学习方式，从下一节开始，我们将从安装到实践，一步步学习 scikit-learn 的基础知识。
