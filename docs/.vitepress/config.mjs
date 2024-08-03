import { defineConfig } from "vitepress";

export default defineConfig({
  title: "Machine Learning",
  titleTemplate: "ML Documentation of SDUT-DMT",
  description: "机器学习课程在线文档",

  lang: "zh-CN",
  base: "/ml/",
  search: {
    provider: "local",
    options: {
      translations: {
        button: {
          buttonText: "搜索文档",
          buttonAriaLabel: "搜索文档",
        },
        modal: {
          noResultsText: "无法找到相关结果",
          resetButtonTitle: "清除查询条件",
          footer: {
            selectText: "选择",
            navigateText: "切换",
          },
        },
      },
    },
  },
  markdown: {
    math: true,
  },

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
        {
          text: "完结与致谢",
          items: [{ text: "完结与致谢", link: "/sklearn/end/thanks" }],
        },
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
            // {
            //   text: "[选读]TensorFlow环境搭建",
            //   link: "/pytorch/env/tensorflow",
            // },
          ],
        },
        {
          text: "深度学习基础",
          items: [
            { text: "初识深度学习", link: "/pytorch/dl/synopsis" },
            { text: "后续内容编写中...", link: "/pytorch/dl/building" },
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
              text: "📙在线Jupyter(代码/笔记)",
              link: "https://www.kaggle.com/code/kqcoxn/20240802-03-sdut-dmt-ml",
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
