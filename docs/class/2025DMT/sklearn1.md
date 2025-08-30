# sklearn 环境搭建

> 介绍 sklearn 的环境搭建

## 关于 Anaconda

Anaconda指的是一个开源的Python发行版本，其中包含了 conda（包管理和环境管理）、Python 等 180 多个科学包及其依赖项。

接下来我们开始安装 Anaconda

## Anaconda 的下载及安装

### 卸载 Python

Anaconda安装之前，应该先检查一下电脑是否安装Python：

- **情况一**：电脑没有安装 Python，或者安装的Python可以卸载；
- **情况二**：电脑已经安装 Python，并且想要保留它；

_此文档仅就情况一进行，情况二的方法较为复杂，最好改为情况一_

#### 验证是否安装 Python

打开命令提示窗口（按下 `Win+R` 打开运行框，输入 `cmd`），在窗口中输入
   
```shell
python --version
```   

若出现了版本号，则说明已经安装了 Python，需要先卸载它。

#### 卸载 Python

在开始菜单中搜索 `控制面板`，然后点击 `程序`，然后点击 `卸载程序`。找到并点击 `Python`，然后点击 `卸载`。

#### 检查删除环境变量

在开始菜单中搜索 `环境变量`来查看，点击 `编辑系统环境变量`。

从 `用户变量`（注意是 **用户变量**）中找到 `path`，选中并点击`编辑`，进入查看自己设置过的环境变量。

将 python 的相关变量全部`删除`，如图中两个值，都选中然后删除，再点击`确定`（若卸载完 python 环境变量自动删掉就不用管了）

![检查删除环境变量](images/检查删除环境变量.png)

在退出时，不要忘记 **点击所有的确认按钮，不要直接叉掉**，否则并没有保存设置。

### 下载 Anaconda 安装包

可以进入 [Anaconda 官网](https://www.anaconda.com/) 进行注册登录后下载，不过因为网络无法访问或者下载速度太慢的问题，更推荐从 [清华大学开源软件镜像站](https://repo.anaconda.com/archive/) 进行下载。

选择22版 `Anaconda3-2022.10-Windows-x86_64.exe` 进行下载，下载后点击 exe 文件，便能够进入安装界面。

### 安装 Anaconda

双击下载好的安装包，点击 `Next`，点击 `I Agree`，选择 `Just Me`，自定义安装位置然后 `Next`。

![安装Anaconda1](images/安装Anaconda1.png)

![安装Anaconda2](images/安装Anaconda2.png)

这个页面中要选择 `Register Anaconda as my default Python 3.x`，**不要**选择 `Add Anaconda to my PATH environment variable`，我们需要后期手动添加环境变量。

![安装Anaconda3](images/安装Anaconda3.png)

点击 `Install`，安装需要等待一会儿，再继续 `Next`。最后页面要将 **两个选项都取消打勾**，点击 `Finish` 完成安装。

![安装Anaconda4](images/安装Anaconda4.png)

安装好后我们需要手动配置环境变量。

### 配置环境变量

>  Anaconda 安装的过程中比较容易出错的环节就是环境变量的配置，所以在配置环境变量的时候要细心一些。

`计算机`（右键）→ `属性` → `高级系统设置` →（点击）`环境变量`

在 `系统变量`（注意是 **系统变量**）里，找到并点击 `Path`。

![配置环境变量1](images/配置环境变量1.png)

在编辑环境变量里，点击 `新建`，输入下面的五个环境变量。

（**这里需要将以下五条环境变量中涉及的到的"C:\ProgramData\Anaconda3"都修改为你的 Anaconda 的安装路径**）

```shell
C:\ProgramData\Anaconda3
C:\ProgramData\Anaconda3\Scripts
C:\ProgramData\Anaconda3\Library\bin
C:\ProgramData\Anaconda3\Library\mingw-w64\bin
C:\ProgramData\Anaconda3\Library\usr\bin
```

![配置环境变量2](images/配置环境变量2.png)

> 简要说明五条路径的用途：这五个环境变量中，1 是 Python 需要，2 是 conda 自带脚本，3 是 jupyter notebook 动态库, 4 是使用 C with python 的时候

新建完成后**一路点击确定**。

### 验证安装

验证 Anaconda 和 python 是否安装成功，win+R 输入 cmd，在弹出的命令行中依次输入 ：

```shell
conda --version
python --version
```

若各自出现版本号，即代表配置成功。

在开始菜单或桌面找到 `Anaconda Navifator` 将其打开（若桌面没有可以发一份到桌面方便后续使用），出现 GUI 界面即为安装成功。

![验证安装](images/验证安装.png)

## 更改Conda源

如果没有 VPN 工具，建议更改 Conda 源来加快下载包的速度。清华大学提供了 Anaconda 的镜像仓库，我们把源改为清华大学镜像源。

找到 Anaconda prompt，打开 shell 面板。

![更改Conda源](images/更改Conda源.png)

在命令行输入以下命令：

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --set show_channel_urls yes
```

验证是否修改好：

```shell
conda config --show channels
```

## 安装 Python 库

Anaconda 自带了一些常用的库，如 numpy、pandas、jupyter、matplotlib、seaborn、scikit-learn 等。

如果需要安装其他库，可以直接在 Anaconda Navigator 里搜索安装。

## VSCode 中指定 Conda 环境

使用 VSCode 进行代码编写，VSCode 支持 conda 环境的切换。

下载 Python 插件并安装，然后在 VSCode 的左下角找到环境选择框，点击右边的设置按钮，选择 conda 环境。

![VSCode指定Conda环境](images/VSCode指定Conda环境.png)

也可以在终端中执行：

```shell
conda activate 环境名称
```

## sklearn运行环境

至此，我们已经在本地配置好了 sklearn 的运行环境，可以用一个简单的例子来测试一下。

新建 Python 文件，输入以下代码并运行：

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据集
iris = load_iris()
# print(iris)
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
print(iris_df.head())

# 可视化数据集
plt.figure(figsize=(10, 6))
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(iris_df[iris_df['species'] == 0]['sepal length (cm)'], iris_df[iris_df['species'] == 0]['sepal width (cm)'], color='red', label='Setosa')
plt.scatter(iris_df[iris_df['species'] == 1]['sepal length (cm)'], iris_df[iris_df['species'] == 1]['sepal width (cm)'], color='green', label='Versicolor')
plt.scatter(iris_df[iris_df['species'] == 2]['sepal length (cm)'], iris_df[iris_df['species'] == 2]['sepal width (cm)'], color='blue', label='Virginica')
plt.legend()
plt.title('Iris Dataset - Sepal Length vs Width')
plt.show()

# 数据预处理
scaler = StandardScaler()
x = scaler.fit_transform(iris.data)

# 对比预处理效果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(iris.data[:, 0], bins=20, color='blue', alpha=0.7)
plt.title('Before Scaling')
plt.subplot(1, 2, 2)
plt.hist(x[:, 0], bins=20, color='green', alpha=0.7)
plt.title('After Scaling')
plt.show()

# 切分数据集
x_train, x_test, y_train, y_test = train_test_split(x, iris.target, test_size=0.3, random_state=42)
print(f"Training set size: {x_train.shape}, {y_train.shape}")
print(f"Testing set size: {x_test.shape}, {y_test.shape}")

# 训练模型
accuracies = []
k_values = range(1, 11)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracies.append(accuracy_score(y_test, y_pred))  # 模型的准确率
print(accuracies)

# 绘制准确率与k值的关系图
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('K Value vs. Accuracy')
plt.grid()
plt.show()

# 确定模型
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# 预测测试集
y_pred = knn.predict(x_test)
for i in range(5):
    print(f"True label: {y_test[i]}, Predicted label: {y_pred[i]}")

# 计算准确率
accurancy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accurancy:.2f}")

# 计算分类报告
report = classification_report(y_test, y_pred)
print(report)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 绘制混淆矩阵热力图
cm_display = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
```

运行后便可以看到相应的结果，具体的使用方法在后续的学习中会掌握。

## 使用 Jupyter

### 使用 Anaconda 启动 Jupyter

Anaconda 中自带了 Jupyter notebook 与 Jupyterlab，可以方便地编写和运行 Python 代码。

使用较新的 Jupyterlab 作为例子，打开 Anaconda Navigator，点击Jupyterlab 下面的 `launch` 按钮（如果没有请先点击 install），Anaconda 会自动在浏览器中打开 Jupyter 界面，关闭页面便会退出环境。

![Anaconda运行Jupyter](images/Anaconda运行Jupyter.png)

关于 Jupyterlab 的使用，可参考[官方文档](https://jupyterlab.readthedocs.io/en/stable/)。

### 使用 VSCode 运行 Jupyter 文件

许多 VSCode  插件对 Jupyter 的运行都进行了支持，例如 Jupyter Extension for Visual Studio Code。

下载并安装插件，新建 `.ipynb` 文件，然后点击右上角的运行按钮，选择运行环境，选择 Anaconda 环境后使用即可。

![VSCode运行Jupyter](images/VSCode运行Jupyter.png)

### 使用 kaggle 在线使用 Jupyter

kaggle 是一个提供数据集和竞赛的平台，我们可以直接在线使用 Jupyter 进行数据分析。

kaggle[官网地址](https://www.kaggle.com/)

首先注册一个 kaggle 账号，然后点击左侧的 `Competitions` 进入竞赛页面，选择一个竞赛，点击 `Join Competition` 进入竞赛页面，点击 `Rules` 进入竞赛规则页面，点击 `Data` 进入数据页面，点击 `Download All` 下载数据集。

点击左侧的 `Notebooks` 进入笔记页面，点击 `New Notebook` 新建笔记，即可使用在线的仿真环境进行数据分析。

## \[附录\] Anaconda 常用命令

在没有 GUI 的情况下，有以下的常用命令：

1.查看当前环境下安装的库：

```shell
conda list
```

2.查看所有环境：

```shell
conda info --envs
```

3.创建新的环境：

```shell
conda create -n 环境名称 python=版本号
```

4.激活环境：

```shell
conda activate 环境名称
```

5.退出环境：

```shell
conda deactivate
```

6.删除环境：

```shell
conda remove -n 环境名称 --all
```

7.导出环境：

```shell
conda env export > environment.yaml
```

8.导入环境：

```shell
conda env create -f environment.yaml
```

9.列出所有可用的包：

```shell
conda search 包名
```

10.安装包：

```shell
conda install 包名
```

11.更新包：

```shell
conda update 包名
```

12.卸载包：

```shell
conda uninstall 包名
```
