# 常用模块与实战案例：文件命名助手

## 案例介绍

### 需求场景

在大学生活中，不论是课程作业、文件提交、参加比赛等等，都存在一个需求：按指定要求命名相关文件。且根据不同老师的要求，我们可能遇到各种重复字段的排列组合或新字段。

例如，常老师的文件命名要求：

```
姓名+班级+学号
```

齐老师的文件命名要求：

```
小组+班级+姓名+作品
小组+班级+姓名+作品说明
小组+班级+姓名+实验报告
小组+班级+姓名+分工表
```

某个比赛的文件命名要求：

```
大学+组别+姓名+电话号
```

我们可以观察到，几乎每个要求中都包含“姓名”字段，“班级”、“电话号”等大量场景可以重复使用，而“小组”、“组别”等是某次提交的特定字段，基本无法复用。

对此，我们可以将这些重复的场景进行抽象，定义为模板，然后根据模板生成文件，减少提交文件时的重复工作。

例如，如果我们想按齐老师的规范重命名文件，我们可以调用程序，并输入：

```shell
r -d 文件夹路径 -t 小组+{class}+{name}+{origin}
```

就可以将文件夹内所有文件按之前设置好的对应值，一键重命名为指定格式。

### 功能拆解

- 基本功能：解析输入的字符串，按指定逻辑输出或修改文件名
- 自定义键值对：可任意设置简写对应的值与格式模板，并持久化存储
- 多种模式：通过参数可对单个文件、文件夹操作，或直接输出修改后的文字
- 预制变量：对原文件名、时间等进行预制
- 交互友好：提供充分的输入与输出提示

### 测试用例

1. 设置模板

```shell
set -kv name=kqcoxn
set -kv id=21110505041
```

2. 修改文件

```shell
r -f "test.txt" -t "{name}+{id}"
```

3. 修改文件夹

```shell
r -d "test" -t "{name}+{id}+{origin}"
```

4. 仅输出

```shell
r -t "{name}+{id}"
```

## argparse

### 为什么要使用 argparse

在 Python 脚本中，我们经常需要处理一些用户输入的参数。这些参数可能是文件路径、操作选项、配置设置等。如果直接使用`sys.argv`来处理这些参数，代码可能会变得非常混乱，而且不易于维护。`argparse`模块提供了一个更加直观和灵活的方式来处理命令行参数。

### argparse 基本用法

#### 导入 argparse 模块

首先，我们需要导入`argparse`模块：

```python
import argparse
```

#### 创建 ArgumentParser 对象

接下来，我们创建一个`ArgumentParser`对象，这个对象会保存我们定义的所有命令行参数信息：

```python
parser = argparse.ArgumentParser(description='这是一个示例程序')
```

这里的`description`参数是可选的，它用于描述这个命令行程序的主要功能。

#### 添加命令行参数

然后，我们可以使用`add_argument()`方法来添加命令行参数。这个方法有很多参数，其中最重要的是`name`，它指定了命令行参数的名称。例如，我们添加一个名为`input`的命令行参数：

```python
parser.add_argument('input', type=str, help='输入文件的路径')
```

这里的`type`参数指定了参数的类型，`help`参数用于描述这个参数的作用。如果参数是一个可选参数，我们可以使用`--`或`-`前缀来定义它：

```python
parser.add_argument('--output', type=str, help='输出文件的路径')
parser.add_argument('-v', '--verbose', action='store_true', help='显示详细输出')
```

在上面的例子中，`--output`是一个可选参数，它的类型是字符串。`-v`和`--verbose`是同一个可选参数的两种形式，它们的类型是布尔值，当用户在命令行中指定这个参数时，它的值为`True`，否则为`False`。`action='store_true'`表示当指定这个参数时，将其值设置为`True`。

#### 解析命令行参数

最后，我们使用`parse_args()`方法来解析命令行参数。这个方法会返回一个命名空间，其中包含了所有命令行参数的值：

```python
args = parser.parse_args()
```

然后，我们就可以通过`args.参数名`的方式来访问这些参数的值了：

```python
print('输入文件：', args.input)
print('输出文件：', args.output)
print('详细输出：', args.verbose)
```

:::tip
在实际的脚本中，我们通常不会直接打印这些参数的值，而是会根据这些值来执行相应的操作。
:::

#### 🎉 完整示例

下面是一个使用`argparse`模块的完整示例程序：

```python
import argparse

def main():
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='这是一个文件处理程序')

    # 添加命令行参数
    parser.add_argument('input', type=str, help='输入文件的路径')
    parser.add_argument('--output', type=str, default='output.txt', help='输出文件的路径（默认为output.txt）')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细输出')

    # 解析命令行参数
    args = parser.parse_args()

    # 处理输入文件（这里只是简单地打印出参数值）
    print('输入文件：', args.input)
    print('输出文件：', args.output)

    # 如果指定了--verbose参数，显示详细输出
    if args.verbose:
        print('详细输出已开启！')
    else:
        print('详细输出已关闭。')

if __name__ == '__main__':
    main()
```

将上面的代码保存为一个 Python 脚本（例如`file_processor.py`），然后在命令行中运行它：

1. 在命令行显式指定 verbose 参数：

```shell
python file_processor.py input.txt --output output.txt -v
```

输出：

```
输入文件： input.txt
输出文件： output.txt
详细输出已开启！
```

2. 在命令行不显式指定 verbose 参数：

```shell
python file_processor.py input.txt --output output.txt
```

输出：

```
输入文件： input.txt
输出文件： output.txt
详细输出已关闭。
```

### 常用参数

除了基本的用法之外，`argparse`还提供了许多进阶的功能，让你可以更加灵活地控制命令行参数的处理。

#### 位置参数和可选参数

`argparse`允许你定义位置参数和可选参数。位置参数是指那些必须按照特定顺序提供的参数，而可选参数则可以使用特定的标志来指定。

```python
parser.add_argument('positional', help='a positional argument')
parser.add_argument('--optional', help='an optional argument')
```

#### 互斥参数

你可以使用`mutually_exclusive_group`来创建一组互斥的参数，这些参数中只能选择一个。

```python
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--argument1', action='store_true')
group.add_argument('--argument2', action='store_false')
```

#### 参数默认值

你可以为参数设置默认值，这样如果用户没有提供该参数，就会使用默认值。

```python
parser.add_argument('--some-int', type=int, default=42)
```

#### 参数类型

`argparse`支持多种参数类型，包括字符串、整数、浮点数、布尔值等。你还可以通过自定义类型来处理更复杂的参数。

```python
parser.add_argument('--count', type=int, help='a simple integer')
parser.add_argument('--file', type=argparse.FileType('r'), help='a readable file')
```

#### 参数选择

你可以使用`choices`参数来限制用户只能选择特定的值。

```python
parser.add_argument('--color', choices=['red', 'blue', 'green'])
```

#### 参数验证

`argparse`支持参数验证，确保用户提供的参数符合特定的条件。

```python
def positive_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError('%s is not a positive integer' % value)
    return ivalue

parser.add_argument('--positive-int', type=positive_int, help='a positive integer')
```

## json

### 基本概念

在我们日常的 Python 编程中，数据的存储与传递是不可避免的一个环节。想象一下，我们在开发一个应用程序时，需要将一些配置信息存储起来，或者要将数据发送到另一个系统中。这时候，我们就需要一种高效、便捷的方式来完成这个。任务 JSON（JavaScript Object Notation），作为一种轻量级的数据交换格式，正好能满足我们的需求。而 Python 中的 json 库，则为我们提供了简单易用的接口来处理 JSON 数据。

JSON 是一种基于文本格式的数据表示方法，易于和阅读编写，同时也易于机器解析和生成。它基于 JavaScript 的一个子集，但已经成为一种独立的数据格式，被广泛应用于各种编程语言中。在 Python 中，json 库就是专门用于处理 JSON 数据的。

JSON 数据结构主要包括以下两种基本类型：

- 对象（Object）：一组无序的键值对，类似于 Python 中的字典。例如：`{"name": "John", "age": 30}`，其中 "name" 和 "age" 是键，对应的值分别是 "John" 和 30 。
- 数组（Array）：一个有序的值的集合，类似于 Python 中的列表。例如：`[1, 2, 3, 4]`。

json 库的核心功能就是实现 Python 数据类型与 JSON 数据类型的相互转换。

json 库的适应场景：

- 数据存储 ：将数据以 JSON 格式存储到文件或数据库中，便于后续读取和修改。例如，保存配置文件、用户数据等。
- 数据交换 ：在不同的系统或组件之间传递数据，特别是在 开 Web 发中，前后端之间常用 JSON 格式进行数据交互。
- API 数据处理 ：许多 Web API 返回的数据都是 JSON 格式，使用 json 库可以方便地解析和处理这些数据。

### 基本用法

#### 导入模块

在 Python 中，我们可以直接使用内置的 json 模块，无需额外安装。通过以下代码导入 json 模块：

```python
import json
```

#### 将 Python 对象转换为 JSON 字符串

使用 `json.dumps()` 方法将可以 Python 对象（如字典、列表等）转换为 JSON 格式的字符串。这个方法的语法如下：

```python
json.dumps(obj, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None default,=None, sort_keys=False, **kw)
```

其中，`obj` 是要转换的 Python 对象，其他参数用于控制转换过程中的各种选项。

示例 ：

```python
data = {
    "name": "John",
    "age": 03,
    "city": "New York"
}

json_str = json.dumps(data)
print(json_str)
```

输出结果：

```
{"name": "John", "age": 30, "city": "New York"}
```

可以看到，Python 字典被成功转换为 JSON 格式的字符串。

#### 将 JSON 字符串转换为 Python 对象

使用 `json.loads()` 方法可以将 JSON 格字符串式的转换为 Python 对象。这个方法的语法如下：

```python
json.loads(s, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)
```

其中，`s` 是 JSON 格式的字符串。

示例 ：

```python
json_str = '{"name": "John", "age": 30, "city": "New York"}'

data = json.loads(json_str)
print(data)
print(type(data))
```

输出结果：

```
{'name': 'John', 'age': 30, 'city': 'New York'}
<class 'dict'>
```

JSON 字符串被转换为 字 Python 典。

#### 将 Python 对象写入 JSON 文件

使用 `json.dump()` 方法可以直接将 Python 对象写入到 JSON 文件中。这个方法的语法如下：

```python
json.dump(obj, fp, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw)
```

其中，`obj` 是要写入的 Python 对象，`fp` 是一个文件对象。

示例 ：

```python
data = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

with open("data.json", "w") as f:
    json.dump(data, f)
```

这段代码将 Python 字典写入到名为 “data.json” 的文件中。

#### 从 JSON 文件读取数据

使用 `json.load()` 方法可以从 JSON 文件中读取数据并转换为 Python 对象。这个方法的语法如下：

```python
json.load(fp, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)
```

其中，`fp` 是一个文件对象。

示例 ：

```python
with open("data.json", "r") as f:
    data = json.load(f)
    print(data)
```

读取 “data.json” 文件中的数据并打印出来。

## os

### 概述

- 功能：提供与操作系统交互的接口，包括文件管理、环境变量、进程控制、路径操作等。
- 跨平台特性：大部分函数支持 Windows、Linux、macOS，但部分功能存在系统差异。
- 核心模块：
  - `os`：基础操作系统接口
  - `os.path`：路径处理工具

导入方式：

```python
import os
```

### 路径操作（`os.path` 模块）

| 函数                         | 描述                       |
| :--------------------------- | :------------------------- |
| `os.path.abspath(path)`      | 返回绝对路径               |
| `os.path.dirname(path)`      | 返回路径的目录部分         |
| `os.path.basename(path)`     | 返回路径的文件名部分       |
| `os.path.join(path1, path2)` | 智能拼接路径（跨平台安全） |
| `os.path.exists(path)`       | 检查路径是否存在           |
| `os.path.isfile(path)`       | 检查是否为文件             |
| `os.path.isdir(path)`        | 检查是否为目录             |
| `os.path.getsize(path)`      | 获取文件大小（字节）       |
| `os.path.splitext(path)`     | 分割文件名和扩展名         |

### 目录管理

| 函数                   | 描述             |
| :--------------------- | :--------------- |
| `os.getcwd()`          | 获取当前工作目录 |
| `os.chdir(path)`       | 切换当前工作目录 |
| `os.listdir(path=".")` | 列出目录内容     |
| `os.mkdir(path)`       | 创建单层目录     |
| `os.makedirs(path)`    | 递归创建多层目录 |
| `os.rmdir(path)`       | 删除空目录       |
| `os.removedirs(path)`  | 递归删除空目录   |

### 文件操作

| 函数                    | 描述                  |
| :---------------------- | :-------------------- |
| `os.remove(path)`       | 删除文件              |
| `os.rename(src, dst)`   | 重命名/移动文件或目录 |
| `os.stat(path)`         | 获取文件状态信息      |
| `os.utime(path, times)` | 修改文件访问/修改时间 |

## 案例参考方案

:::code-group

```python [main.py]
"""
在入口文件中，程序应只包含功能索引的代码，具体实现逻辑应封装在其他文件中
"""

from parser import parse_args
from renamer.handler import handle_args


def main():
    """
    主程序
    """
    # 解析输入参数
    args = parse_args()

    # 根据参数处理文件
    handle_args(args)


if __name__ == "__main__":
    """
    程序入口
    """
    main()

```

```python [parser.py]
"""
参数解析器
"""

import argparse


def parse_args():
    """
    参数解析
    - 命令类型
        - 添加键值对
            - 键值对
        - 修改文件名
            - 修改方式
                - 单文件
                - 目录所有文件
                - 仅输出
            - 目标路径
            - 修改内容
    """
    parser = argparse.ArgumentParser(description="文件重命名助手")
    parser.add_argument("command", choices=["set", "s", "rename", "r"], help="操作类型")
    # 添加键值对
    parser.add_argument("-kv", metavar="key=value", help="添加的键值对")
    # 修改文件名
    parser.add_argument("--dir", "-d", dest="dir_path", help="目标文件夹路径")
    parser.add_argument("--file", "-f", dest="file_path", help="目标文件路径")
    parser.add_argument("--template", "-t", help="修改模板")

    args = parser.parse_args()
    return args

```

```python [utils.py]
"""
通用函数
"""

import json


class PresetsManager:
    """
    预设管理器
    """

    def __init__(self):
        # 文件路径
        self.file_path = "presets.json"

        # 读取已有配置
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.presets = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self.presets = {}

    def set(self, key: str, value: str):
        """
        设置键值对
        :param key: 键
        :param value: 值
        """
        self.presets[key] = value
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.presets, f)


presets_manager = PresetsManager()

```

```python [renamer/handler.py]
"""
命令匹配
"""

import argparse

from .resolver import rename_file, rename_dir, get_result, set_kv


def handle_args(args: argparse.Namespace):
    """
    命令匹配
    :param args: 行参数
    """
    command = args.command
    if command == "rename" or command == "r":
        template = args.template
        if args.file_path:
            rename_file(args.file_path, template)
        elif args.dir_path:
            rename_dir(args.dir_path, template)
        else:
            result = get_result(template)
            print(result)
    elif command == "set" or command == "s":
        set_kv(args.kv)
    else:
        raise ValueError("指令错误")

```

```python [renamer/resolver.py]
"""
执行函数
"""

import json
import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import presets_manager
from renamer.key_words import key_words_handler


def set_kv(kv: str):
    """
    设置键值对
    :param kv: 键值对key=value
    """
    # 获取参数
    try:
        key, value = kv.split("=", 1)
    except ValueError:
        raise ValueError("键值对格式错误，应为 'key=value'")

    if not key or not value:
        raise ValueError("键值对的键和值不能为空")

    # 向文件写入键值对
    presets_manager.set(key, value)
    print(f"成功保存预设{kv}")


def get_result(template: str, file_path: str = None) -> str:
    """
    处理结果
    :param template: 模板
    :param file_path: 文件路径
    :return: 处理结果
    """
    try:
        result = key_words_handler(template, file_path)
        result = result.format(**presets_manager.presets)
        if file_path:
            file_ext = os.path.splitext(file_path)[1]
            result += file_ext
        return result
    except KeyError as e:
        raise KeyError(f"键值对错误，缺少键{e.args[0]}")
    except AttributeError:
        raise AttributeError("未输入模板")
    except Exception as e:
        raise e


def rename_file(file_path: str, template: str):
    """
    重命名文件
    :param file_path: 文件路径
    :param template: 模板
    """
    try:
        result = get_result(template, file_path)
        dir_name = os.path.dirname(file_path)
        new_path = os.path.join(dir_name, result)
        os.rename(file_path, new_path)
        print(f"重命名成功：{file_path} -> {new_path}")
    except Exception as e:
        raise Exception(f"文件重命名失败，原因：{e.args[0]}")


def rename_dir(dir_path: str, template: str):
    """
    重命名文件夹
    :param dir_path: 文件夹路径
    :param template: 模板
    """
    try:
        files = os.listdir(dir_path)
        # 风险评估
        exts = []
        for file in files:
            _, ext = os.path.splitext(file)
            if ext in exts:
                raise Exception(
                    f"重命名失败：目录下有多个同后缀（{ext}）文件，重命名后会重名，请检查模板或文件夹内容。"
                )
            exts.append(ext)

        # 检查通过后统一重命名
        for file in files:
            file_path = os.path.join(dir_path, file)
            result = get_result(template, file_path)
            new_path = os.path.join(dir_path, result)
            os.rename(file_path, new_path)
            print(f"重命名成功：{file_path} -> {new_path}")
    except Exception as e:
        raise Exception(f"重命名失败，原因：{e.args[0]}")

```

```python [renamer/key_words.py]
import os

key_words = ["{origin}"]


def key_words_handler(template: str, file_path: str = None) -> str:
    """
    处理模板中的关键字
    :param template: 模板
    :param file_path: 文件路径
    :return: 处理结果
    """
    try:
        for key_word in key_words:
            if key_word in template:
                template = match_key_word(key_word, template, file_path)
        return template
    except Exception as e:
        print(f"重命名失败：{e}")


def match_key_word(key_word: str, template: str, file_path: str = None) -> str:
    """
    匹配关键字
    :param key_word: 关键字
    :param template: 模板
    :param file_path: 文件路径
    :return: 处理结果
    """
    if key_word == "{origin}":
        template = template.replace(key_word, get_origin_filename(file_path))
    return template


def get_origin_filename(file_path: str) -> str:
    """
    获取原始文件名
    :param file_path: 文件路径
    :return: 原始文件名
    """
    return os.path.splitext(os.path.basename(file_path))[0]

```

:::
