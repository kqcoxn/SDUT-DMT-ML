# 批量归一化

深层神经网络的训练，尤其是使网络在较短时间内收敛是十分困难的，批量归一化（batch normalization）是一种流行且有效的技术，能加速深层网络的收敛速度，目前仍被广泛使用。

## 训练深层网络时的问题

![](../images/cnn/deep_model.png)

深度神经网络在训练时会遇到一些问题：

- **收敛速度慢**：由于训练时先正向传播后反向传播，且每层的梯度一般较小，若网络较深，则反向传播时会出现类似于梯度消失的现象，导致距离数据更近的层梯度较小，收敛慢，而距离输出更近的层梯度较大，收敛快。然而底部的层一般都用于提取较基础的特征信息，上方的层收敛后，由于底部提取基础特征的层仍在变化，上方的层一直在不停的重新训练，导致整个网络难以收敛，训练较慢。
- **内部协变量转移**：
  - **分布偏移**：偏移在视频课程中并未出现，但在《动手学深度学习》这本书中有提到过，在 4.9. 环境和分布偏移部分。偏移指的是训练数据可能和测试数据的分布不同，比如利用来自真实的猫和狗的照片的训练数据训练模型，然后让模型去预测动画中的猫和狗的图片。这显然会降低正确率也会对模型的进一步优化带来干扰。一般情况下对于分布偏移我们毫无办法，然而，在一些特定场景中，如果假定一些训练数据和测试数据分布的前提条件，就能对分布偏移进行处理，其中之一就是协变量偏移。
    ![](../images/cnn/cat-dog-train.svg)
  - **协变量偏移**：协变量偏移假设输入的分布可能随时间变化，但标签函数（条件分布 $P(y|\boldsymbol{x})$）没有改变。统计学家称这为协变量偏移（covariate shift）并给出了一些解决方案。
  - **内部协变量偏移**(Internal Covariate Shift)：每一层的参数在更新过程中，会改变下一层输入的分布，导致网络参数变幻莫测，难以收敛，神经网络层数越多，表现得越明显。
- **过拟合**：由于网络深度加深，变得更为复杂，使得网络容易过拟合。

## 批量归一化

批量归一化(batch normalization)在 [Ioffe & Szegedy, 2015]中被提出，用于解决上述训练深度网络时的这些问题，然而这只是人们的感性理解，关于批量归一化具体是怎样帮助训练这个问题目前仍待进一步研究。

批量归一化尝试将每个训练中的 mini-batch 小批量数据（即会导致参数更新的数据）在每一层的结果进行归一化，使其更稳定，归一化指的是对于当前小批量中的所有样本，求出期望和方差，然后将每个样本减去期望再除以标准差。

## 形式化表达

下面的运算均为向量运算，向量中的每个维度代表一个特征，对于每个特征分别进行计算再拼接在一起即为向量运算:

设 $\mathbf{x} \in \mathcal{B}$ 为来自一个小批量 $\mathcal{B}$ 的输入，批量规范化 BN 根据下式进行转换：

$$
\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}}}{\hat{\boldsymbol{\sigma}}_{\mathcal{B}}} + \boldsymbol{\beta}.
$$

式中 $\hat{\boldsymbol{\mu}}_{\mathcal{B}}$ 为小批量 $\mathcal{B}$ 样本均值，$\hat{\boldsymbol{\sigma}}_{\mathcal{B}}$ 为样本标准差：

$$
\begin{aligned}
\hat{\boldsymbol{\mu}}_{\mathcal{B}} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}, \\
\hat{\boldsymbol{\sigma}}_{\mathcal{B}}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon,
\end{aligned}
$$

其中 $\epsilon$ 用于防止分母为 0。经过减期望与除以标准差后得到期望为 0 方差为 1 的小批量数据。为了使小批量有更自由的选择，再将其乘以拉伸参数 $\boldsymbol{\gamma}$，加上偏移参数 $\boldsymbol{\beta}$。这两个参数与 $\boldsymbol{x}$ 同样大小，是模型中的可学习参数，与其他参数一同更新。

由于 $\hat{\boldsymbol{\mu}}_{\mathcal{B}}$ 和 $\hat{\boldsymbol{\sigma}}_{\mathcal{B}}$ 是由当前小批量计算的值，实际上是整个分布对应的期望与标准差的估计值。由于小批量的随机选择，$\hat{\boldsymbol{\mu}}_{\mathcal{B}}$ 和 $\hat{\boldsymbol{\sigma}}_{\mathcal{B}}$ 会给模型带来一定的与输入数据有关的噪音，而这些噪音也能对模型进行正则化，防止过拟合。为何这种噪音能加快训练并带来正则化还有待研究，不过已有理论说明了为什么批量规范化最适应 $50∼100$ 范围中的中等批量大小的问题。

训练时不能使用整个数据集，只能一步步的训练和更新；而预测时模型已然固定，可以根据整个数据集精确计算均值和方差。因此，批量归一化对于训练和预测时有两种不同模式。

## 批量归一化层

批量归一化不再单独的考虑单个样本，需要对整个 mini-batch 进行，因此需要考虑多种情况。

### 全连接层

通常，我们将批量规范化层置于全连接层中的仿射变换和激活函数之间。如下：

$$
\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}))
$$

### 卷积层

在卷积层中，我们将通道视作每个位置的特征，将每个样本中的每个位置视作一个样本进行计算。每个通道都有着自己的拉伸参数 $\gamma$ 和偏移参数 $\beta$，所有通道加在一起组成了拉伸参数向量 $\boldsymbol{\gamma}$ 和偏移参数向量 $\boldsymbol{\beta}$。若样本数为 $m$，卷积输出为 $p \times q$，计算时对 $m \times p \times q$ 个向量进行批量归一化运算（即视作有 $m \times p \times q$ 个样本）。

### 预测过程中的批量归一化

在训练过程中，我们需要不断地更新模型，方差和均值也就在不断地变化，就必须计算当前小批量数据对应的方差和均值，然而预测时我们的模型已经确定下来，可以用在整个训练数据集上得到的均值和方差来对预测时的结果进行归一化。

### 实现细节

在实际实现时，一般使用指数加权平均来更新小批量的均值和方差，指数加权平均将旧值和当前计算结果不断进行加权平均，最终做到平滑的向更新值靠拢，公式如下：

$$
S_t =
\begin{cases}
Y_1, & t = 1 \\
\beta S_{t-1} + (1 - \beta) Y_t, & t > 1
\end{cases}
$$

批量归一化的参数可以通过动量梯度下降，RMSProp，Adam 等多种优化方法进行训练。

## \[选读\]吴恩达老师深度学习课程中的批量归一化

吴恩达老师深度学习课程中的批量归一化中的部分内容与本课程有所出入，考虑到批量归一化这部分内容还没有精确的理论解释，目前的认识仅限于直觉，故将两课程中的区别即补充罗列在此作为参考：

- 关于 dropout：
  - 本课中提到批量归一化有正则化效果，无需再进行 dropout
  - 吴恩达老师课程中提到批量归一化正则化效果较差，不能作为正则化的手段，必要时需要 dropout
- 对于线性层（包括其他带有偏置项的层）后的批量归一化，由于归一化时减去了均值，偏置项被消掉，可以省略归一化层之前的偏置项
- 标准化的输入能使梯度下降加快，批归一化能使得每层的输入都被归一化，这也是训练更快的原因之一
- 批量归一化可以使得不同层之间互相的影响减少，从而应对数据偏移，增强鲁棒性。
