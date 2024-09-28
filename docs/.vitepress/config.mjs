import { defineConfig } from "vitepress";
import { SearchPlugin } from "vitepress-plugin-search";

export default defineConfig({
  title: "Machine Learning",
  titleTemplate: "ML Documentation of SDUT-DMT",
  description: "机器学习课程在线文档",

  lang: "zh-CN",
  base: "/ml/",
  markdown: {
    math: true,
  },

  plugins: [
    SearchPlugin({
      encode: false,
      tokenize: "full",
    }),
  ],

  head: [
    ["link", { rel: "icon", href: "/sklearn/logo.png" }],
    ["link", { rel: "stylesheet", href: "/sklearn/custom.css" }],
  ],

  themeConfig: {
    logo: "/logo.png",
    outlineTitle: "本页大纲",
    outline: [2, 3],

    nav: [
      { text: "首页", link: "/index" },
      { text: "ML&Sklearn", link: "/sklearn/synopsis/synopsis" },
      { text: "DL&Pytorch", link: "/pytorch/synopsis/synopsis" },
      { text: "实验", link: "/exp/synopsis/synopsis" },
      { text: "贡献者", link: "/team" },
    ],

    sidebar: {
      "/sklearn/": [
        {
          text: "概述",
          items: [
            { text: "简介", link: "/sklearn/synopsis/synopsis" },
            { text: "准备工作", link: "/sklearn/synopsis/prepare" },
            { text: "文档说明", link: "/sklearn/synopsis/doc" },
            {
              text: "[选项1]解释器环境搭建",
              link: "/sklearn/synopsis/env",
            },
            {
              text: "[选项2]Anaconda环境搭建",
              link: "/sklearn/synopsis/anaconda",
            },
          ],
        },
        {
          text: "Python基础",
          items: [
            { text: "Python入门", link: "/sklearn/python/base" },
            { text: "NumPy", link: "/sklearn/python/numpy" },
            { text: "Matplotlib", link: "/sklearn/python/matplotlib" },
            { text: "Pandas", link: "/sklearn/python/pandas" },
          ],
        },
        {
          text: "RE:机器学习",
          items: [
            { text: "初识人工智能", link: "/sklearn/intro/ai" },
            { text: "机器学习概述", link: "/sklearn/intro/ml" },
            { text: "[选读]深度学习简介", link: "/sklearn/intro/dl" },
          ],
        },
        {
          text: "线性回归模型",
          items: [
            { text: "线性回归", link: "/sklearn/linear/synopsis" },
            { text: "API初步使用", link: "/sklearn/linear/api1" },
            { text: "损失与优化", link: "/sklearn/linear/loss" },
            { text: "[选读]梯度下降实现", link: "/sklearn/linear/gradient" },
            { text: "再谈API", link: "/sklearn/linear/api2" },
            { text: "[案例]波士顿房价预测", link: "/sklearn/linear/example" },
            { text: "欠拟合与过拟合", link: "/sklearn/linear/fitting" },
            { text: "[选读]正则化线性模型", link: "/sklearn/linear/regular" },
            { text: "岭回归改进", link: "/sklearn/linear/ridge" },
            { text: "模型的保存和加载", link: "/sklearn/linear/model" },
          ],
        },
        {
          text: "我推的机器学习",
          items: [
            { text: "拆分数据集", link: "/sklearn/skills/resolution" },
            { text: "偏差与方差", link: "/sklearn/skills/varies" },
            { text: "特征预处理", link: "/sklearn/skills/preprocess" },
            { text: "项目技巧", link: "/sklearn/skills/project" },
            { text: "[选读]倾斜数据集处理", link: "/sklearn/skills/skew" },
          ],
        },
        {
          text: "逻辑回归模型",
          items: [
            { text: "逻辑回归", link: "/sklearn/logistic/synopsis" },
            { text: "API使用", link: "/sklearn/logistic/api" },
            {
              text: "[案例]癌症分类预测",
              link: "/sklearn/logistic/example",
            },
            { text: "分类评估", link: "/sklearn/logistic/evaluation" },
            { text: "[选读]绘制ROC曲线", link: "/sklearn/logistic/roc" },
          ],
        },
        {
          text: "K-临近算法",
          items: [
            { text: "KNN", link: "/sklearn/knn/synopsis" },
            { text: "[选读]距离度量", link: "/sklearn/knn/distance" },
            { text: "API使用", link: "/sklearn/knn/api" },
            { text: "[选读]kd树", link: "/sklearn/knn/kdtree" },
            { text: "[案例]鸢尾花种类预测", link: "/sklearn/knn/example" },
          ],
        },
        {
          text: "[选读]神经网络初步",
          items: [
            { text: "神经网络", link: "/sklearn/neural/synopsis" },
            { text: "更多激活函数", link: "/sklearn/neural/activation" },
            { text: "多分类问题", link: "/sklearn/neural/softmax" },
            { text: "更加高级的神经网络", link: "/sklearn/neural/plus" },
          ],
        },
        {
          text: "决策树模型",
          items: [
            { text: "决策树算法", link: "/sklearn/tree/synopsis" },
            { text: "cart剪枝", link: "/sklearn/tree/cart" },
            { text: "API使用", link: "/sklearn/tree/api" },
            { text: "特征提取", link: "/sklearn/tree/trail" },
            {
              text: "[案例]泰坦尼克号乘客生存预测",
              link: "/sklearn/tree/example",
            },
            { text: "[选读]可视化API", link: "/sklearn/tree/visualize" },
          ],
        },
        {
          text: "集成学习初步",
          items: [
            { text: "集成学习", link: "/sklearn/ensemble/synopsis" },
            { text: "Bagging", link: "/sklearn/ensemble/bagging" },
            { text: "随机森林API", link: "/sklearn/ensemble/forest" },
            { text: "Boosting", link: "/sklearn/ensemble/boosting" },
          ],
        },
        {
          text: "K-means聚类模型",
          items: [
            { text: "聚类算法", link: "/sklearn/kmeans/cluster" },
            { text: "K-means算法", link: "/sklearn/kmeans/synopsis" },
            { text: "API使用", link: "/sklearn/kmeans/api" },
            { text: "模型评估", link: "/sklearn/kmeans/evaluation" },
            { text: "[选读]算法优化", link: "/sklearn/kmeans/optimization" },
          ],
        },
        {
          text: "异常检测算法",
          items: [
            { text: "异常检测", link: "/sklearn/anomaly/synopsis" },
            { text: "API使用", link: "/sklearn/anomaly/api" },
            { text: "[选读]特征选择原则", link: "/sklearn/anomaly/trail" },
          ],
        },
        {
          text: "降维算法",
          items: [
            { text: "降维", link: "/sklearn/reduction/synopsis" },
            { text: "特征选择", link: "/sklearn/reduction/trail" },
            { text: "主成分分析(PCA)", link: "/sklearn/reduction/pca" },
            { text: "[案例]用户喜好分析", link: "/sklearn/reduction/example" },
          ],
        },
        {
          text: "支持向量机",
          items: [
            { text: "SVM", link: "/sklearn/svm/synopsis" },
            { text: "线性可区分问题", link: "/sklearn/svm/linear" },
            { text: "核方法概述", link: "/sklearn/svm/kernel" },
            { text: "API使用", link: "/sklearn/svm/api" },
          ],
        },
        {
          text: "[已删除]推荐系统初步",
          items: [
            {
              text: "[已删除]推荐系统",
              link: "/sklearn/recommend/synopsis",
            },
          ],
        },
        {
          text: "[选读]强化学习初步",
          items: [
            { text: "强化学习", link: "/sklearn/reinforce/synopsis" },
            {
              text: "离散状态空间的强化学习",
              link: "/sklearn/reinforce/discrete",
            },
            {
              text: "[已删除]连续状态空间的强化学习",
              link: "/sklearn/reinforce/continuous",
            },
          ],
        },
        // {
        //   text: "完结与致谢",
        //   items: [{ text: "完结与致谢", link: "/sklearn/end/thanks" }],
        // },
      ],
      "/pytorch/": [
        {
          text: "概述",
          items: [
            { text: "简介", link: "/pytorch/synopsis/synopsis" },
            { text: "准备工作", link: "/pytorch/synopsis/prepare" },
          ],
        },
        {
          text: "环境搭建",
          items: [
            { text: "Pytorch环境搭建指南", link: "/pytorch/env/guide" },
            {
              text: "[选项1]GPU版Pytorch搭建",
              link: "/pytorch/env/gpu",
            },
            {
              text: "[选项2]CPU版Pytorch搭建",
              link: "/pytorch/env/cpu",
            },
          ],
        },
        {
          text: "预备知识",
          items: [
            { text: "初识深度学习", link: "/pytorch/dl/synopsis" },
            { text: "数据操作API", link: "/pytorch/dl/data" },
            { text: "自动求导与梯度计算", link: "/pytorch/dl/autograd" },
          ],
        },
        {
          text: "线性神经网络",
          items: [
            { text: "再看线性回归", link: "/pytorch/linear/linear" },
            { text: "Softmax回归", link: "/pytorch/linear/softmax" },
          ],
        },
        {
          text: "深度学习基础",
          items: [
            { text: "多层感知机", link: "/pytorch/base/perceptron" },
            { text: "丢弃法", link: "/pytorch/base/dropout" },
            { text: "数值稳定性", link: "/pytorch/base/numerical" },
            { text: "神经网络API", link: "/pytorch/base/api" },
          ],
        },
        {
          text: "卷积神经网络",
          items: [
            { text: "卷积层", link: "/pytorch/cnn/conv" },
            { text: "填充和步幅", link: "/pytorch/cnn/padding" },
            { text: "多输入输出通道", link: "/pytorch/cnn/channel" },
            { text: "池化层", link: "/pytorch/cnn/pool" },
            { text: "经典卷积神经网络LeNet", link: "/pytorch/cnn/lenet" },
            { text: "AlexNet", link: "/pytorch/cnn/alexnet" },
            { text: "使用块的网络VGG", link: "/pytorch/cnn/vgg" },
            { text: "网络中的网络NiN", link: "/pytorch/cnn/nin" },
            { text: "GoogLeNet", link: "/pytorch/cnn/googlenet" },
            { text: "批量归一化", link: "/pytorch/cnn/batchnorm" },
            { text: "残差网络ResNet", link: "/pytorch/cnn/resnet" },
            { text: "数据增广", link: "/pytorch/cnn/dataaug" },
            { text: "微调", link: "/pytorch/cnn/finetune" },
          ],
        },
        {
          text: "[选读]计算机视觉(CV)初步",
          items: [
            { text: "物体检测", link: "/pytorch/cv/detection" },
            { text: "语义分割", link: "/pytorch/cv/segmentation" },
          ],
        },
        {
          text: "循环神经网络",
          items: [
            { text: "序列模型", link: "/pytorch/rnn/seqmodel" },
            { text: "语言模型", link: "/pytorch/rnn/langmodel" },
            { text: "循环神经网络RNN", link: "/pytorch/rnn/rnn" },
            { text: "门控循环单元GRU", link: "/pytorch/rnn/gru" },
            { text: "长短期记忆网络LSTM", link: "/pytorch/rnn/lstm" },
            { text: "深层循环神经网络", link: "/pytorch/rnn/deep" },
            { text: "编码器-解码器架构", link: "/pytorch/rnn/encoder" },
            { text: "序列到序列学习", link: "/pytorch/rnn/seq2seq" },
            { text: "束搜索", link: "/pytorch/rnn/beamsearch" },
          ],
        },
        {
          text: "[选读]注意力机制概述",
          items: [
            { text: "注意力机制", link: "/pytorch/attention/synopsis" },
            { text: "Transformer", link: "/pytorch/attention/transformer" },
          ],
        },
        {
          text: "[已删除]自然语言处理初步",
          items: [
            { text: "[已删除]自然语言处理", link: "/pytorch/nlp/synopsis" },
          ],
        },
        {
          text: "[选读]优化算法概述",
          items: [{ text: "常见的优化算法", link: "/pytorch/optim/synopsis" }],
        },
        {
          text: "[选读]高性能计算",
          items: [
            { text: "深度学习硬件", link: "/pytorch/hpc/hardware" },
            { text: "单击多卡并行", link: "/pytorch/hpc/parallel" },
            { text: "分布式训练", link: "/pytorch/hpc/distributed" },
          ],
        },
      ],
      "/exp/": [
        {
          text: "概述",
          items: [{ text: "简介", link: "/exp/synopsis/synopsis" }],
        },
        {
          text: "数媒暑期知识补充课程(2024)",
          items: [
            { text: "简介", link: "/exp/SCForDMT/synopsis" },
            { text: "Python基础", link: "/exp/SCForDMT/python" },
            { text: "数据库相关知识", link: "/exp/SCForDMT/db" },
            { text: "机器学习基础知识", link: "/exp/SCForDMT/ml" },
            { text: "课后习题", link: "/exp/SCForDMT/homework" },
            {
              text: "📙在线Jupyter(笔记/代码)",
              link: "https://www.kaggle.com/code/kqcoxn/20240802-03-sdut-dmt-ml",
            },
          ],
        },
        {
          // text: "SciKit-Learn 先导课",
          text: "sklearn环境搭建",
          items: [
            { text: "简介", link: "/exp/sklearnIL/synopsis" },
            { text: "在线文档", link: "/exp/sklearnIL/env" },
            { text: "完整课程", link: "/exp/sklearnIL/handout" },
            { text: "📊在线PPT", link: "https://kdocs.cn/l/cdWbNfF9HlSF" },
            {
              text: "📙在线Jupyter(讲义/代码)",
              link: "https://www.kaggle.com/code/kqcoxn/sdut-dmt-ml-scikit-learn-il",
            },
          ],
        },
        {
          text: "实验一：回归预测",
          items: [
            { text: "简介", link: "/exp/regression/synopsis" },
            {
              text: "项目一：预测患者糖尿病病情",
              link: "/exp/regression/proj_1",
            },
            {
              text: "项目五：股票价格走势预测",
              link: "/exp/regression/proj_5",
            },
            {
              text: "📙在线Jupyter(思路/代码)",
              link: "https://www.kaggle.com/code/kqcoxn/sdut-dmt-ml-scikit-learn-e1p1",
            },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/kqcoxn/SDUT-DMT-ML" },
    ],
  },
});
