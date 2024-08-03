# Python 入门

~~本节又名：《5 分钟入门不了 Python》~~

> 对于本章内容，如果你已经有 Python 相关的基础可以直接跳过，如果对 numpy、pandas 等库仍不熟悉，可以当作快速笔记使用。本节内容只涉及 Python 的面向过程部分，面向对象部分如有需要请自行补足。

## Python

### 简介

Python 是一门程序设计语言。在开发者眼里，语言大致可以分为 3 类：

- **自然语言**：人能听懂的语言，例如汉语，英语，法语等等。
- **机器语言**：机器能听懂的语言，机器只能听懂 `0` 和 `1`。
- **程序设计语言**：机器能够听懂，人能听懂的语言，例如 `Python`，`C`，`C++` 、`C#` 、`Java`、`js` 等等。

同时，Python 是一门**解释型语言**，这意味着你在运行程序的时候，不需要先把代码编译成机器语言，而是直接运行解释器，解释器会把代码一行一行地解释执行。（所以 Python 经常会被吐槽跑程序很慢）

### 特点

Python 的特点主要有语法简洁、类库强大、胶水语言（调用其他语言的类库）、代码量较少等特点，在代码实践当中你会明显的感到 Python 与 C 等其他语言的不同。

此外，Python 主要有以下优点：

- **入门简单**：Python 语法简单，易于学习，学习曲线平滑。
- **丰富的库**：Python 自带的库非常丰富，涵盖了数据处理、Web 开发、科学计算等领域。
- **跨平台**：Python 可以运行于各种平台，包括 `Windows`、`Linux`、`MacOS` 等。

### 安装

请参考[环境搭建](../synopsis/env.html#python)部分

## 基础知识

在用 Python 写代码的之前，对 Python 的基础知识是必须要会的，不然你可能会寸步难行。基础知识包括**输入输出、变量、数据类型、表达式、运算符**这 5 个方面。

### 输入输出

Python 有很多函数，后面我们会细讲，但这里先将两个最基本的函数：输入和输出。

输出函数 `print()`：

```python
print(要输出的内容)
```

输入函数是 `input()`，功能是接收用户输入的内容，语法是：

```python
输入的内容 = input(提示信息)
```

举例：接收用户输入的密码并打印出来：

```python
n = input("请输入密码：")	#把输入内容赋给n，用 n 接收一下
print(n)	#打印n
```

在 Python 里，`#` 表示注释，“#”后面的东西不会被执行。代码运行之后首先出现了`请输入密码：`，然后随意输入，比如输入 `123`，执行结果：

```shell
提示信息
请输入密码：123
123
```

### 变量

变量就是一个名字，需要先赋值在使用，变量要符合标识符(名字)的命名规范，这是硬性要求，标识符相当于名字，包括变量名、函数名、类名等等，

标识符的命名规则如下：

- 合法的标识符:**字母，数字(不能开头),下划线**，py3 可以用中文（不建议），py2 不可以。
- **大小写**敏感。
- 不能使用**关键字**和**保留字**。
- 没有长度限制。

关于命名规范，本文不进行深入讨论，有兴趣可以参考[PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)。

### 数据类型

#### 分类

数据类型可分为以下 6 类：

1. **整型**：整数，英文名 `int` ，例如 `5` 的数据类型就是整型。
2. **浮点型**：小数，英文名 `float` ，例如 `0.5` 就是 1 个浮点型数据。也可以用科学计数法，e 表示乘以 10 几次方，例如 `b=1e10` 表示 1\*10 的 10 次方。
3. **字符串**：英文 `str`，表现形式有 4 种：`'xs'` 、 `"xs"` 、 `"""xsxs"""` 、 `''''xxx'''` ，三引号有个特殊功能，表示注释，跟 `#` 一样的功能。
4. **布尔类型**：英文 `bool`，`True` 为真，`False` 为假；`1` 表示真，`0` 表示假。
5. `None` 是一个单独的数据类型。
6. **列表**、**元组**、**字典**、**集合**也是常见的数据类型（复杂数据类型）。

在写代码的时候我们经常需要将不同的数据类型进行转换，主要的数据类型转换方法如下：

- 字符串转整型：`int(str)`
- 浮点型转整型：`int(float)`
- 字符串转浮点型：`float(str)`
- 整型转字符串：`str(int)`
- 布尔类型转整型：`int(bool)`
- 整型转布尔类型：`bool(int)`

例如：

```python
f = 30
ff = float(f)  # 30.0
print(ff)
```

#### 类型判断

常用的获取数据类型信息的函数有 `type()`和 `isinstance()`，例如：

```python
f = 30
print(type(f))
n = isinstance(f,int)   #用n来接收一下结果
print(n)
```

输出结果：

```shell
<class 'int'>
False
```

### 表达式

在 Python 中，表达式是由数字、算符、数字分组符号（括号）、变量等对象的组合叫做表达式，表达式均有固定字面值，例如 `10+20`这个表达式的值为 `30`，表达式 `10>30` 的值为 `False` 。

### 运算符

运算符可以分为 4 类：

- 一般运算符：`+`、`-`、`*`、`/`、`%`、`**` 等。
- 赋值运算符：`=`、`+=`、`-=`、`*=`、`/=`、`%=`、`**=` 等。
- 比较运算符：`==`、`!=`、`>`、`>=`、`<=`、`is`、`is not` 等。
- 逻辑运算符：`and`、`or`、`not` 等。

例如：

```python
a = 10
b = 20
c = 30
d = 40
n1 = a > b and a < c    #a>b为假，a<c为真，假与真为假
n2 = not a < c   #a<c为真，非真则为假
n3 = a > b or a < c     #a>b为假，a<c为真，假或真为真
print(n1,n2,n3)
```

输出结果：

```shell
False False True
```

### 缩进

Python 是一种缩进语言，即代码块的缩进表示代码块的开始。Python 代码块的缩进必须是 `4` 个空格（大多数 IDE 中使用 `Tab` 键即可），不能使用制表符。代码块的结束是由缩进的结束来表示，不需要其他的符号。

例如：

```python
if a > b:
    print("a大于b")
else:
    print("a不大于b")
```

### 导入库

1. 导入单个库：`import 库名`
2. 导入多个库：`import 库1,库2,库3`
3. 导入指定函数：`from 库名 import 函数名`
4. 导入所有函数：`from 库名 import *`

例如：

```python
import math
import random
from math import sqrt
from random import randint

print(math.pi)
print(random.randint(1, 100))
print(sqrt(9))
```

输出结果：

```shell
3.141592653589793
12
3.0
```

## 流程控制

流程控制是指根据条件执行不同的代码块，Python 提供了 `if-else`、`for/while` 等流程控制语句。

### 条件分支流程

当达到某种条件的时候才会触发的代码。

语法：

```python
s = int(input("请输入分数:"))
if 80 >= s >= 60:
    print("及格")
elif 80 < s <= 90:
    print("优秀")
elif 90 < s <= 100:
    print("非常优秀")
else:
    print("不及格")
    if s > 50:
        print("你的分数在60分左右")
    else:
        print("你的分数低于50分")
```

输入输出：

```shell
请输入分数:55
不及格
你的分数在60分左右
```

### 循环流程

循环语句的作用就是重复运行某些代码。

#### while 循环

语法：

```python
# 请输入一个整数，并计算各个位和 如：321=6
n = int(input("请输入一个整数:"))  # 将字符串转为整型
# sums累加器：m=10 m=10+5
sums = 0
while n != 0:  # 32 #3
    sums = sums + n % 10  # sums=1+2=3+3=6
    n = n // 10  # 32
print(sums)
```

输入输出：

```shell
请输入一个整数:2345
14
```

#### for 循环

for 循环和 while 循环都是循环语句，但不一样的点在于 for 循环是计数循环。

语法：

```python
l=[3,2,1]
for n in l:
	print("1")
```

输出结果：

```shell
1
1
1
```

其中，`l` 是个列表，后面我们会讲，列表里面有 3 个元素，每执行一次 `for` 循环，列表里面的元素就会被赋值给 `n`，直到列表里面没有了元素可赋值，则 `n` 就跳出了列表，此时的 for 循环就不成立了，不执行 `for` 里面的代码块。

#### range() 函数

for 循环经常会搭配 `range` 来使用，`range` 是一个可迭代对象，其声明如下：

```python
range(start=0,stop,step=1)
```

其中：

- `start` 值的是开始下标。range 序列里面的所有元素都有下标，默认从 `0` 开始。
- `stop` 是结束位置。结束的位置下标为（元素个数-1），例如 range 里面有 `4` 个元素，那么结束下标最大为 `3`,大于 3 则跳出 range。
- `step` 是步长，如果 step 是 `2`，那么每次会隔开 `1` 个元素。默认步长为 `1`，即每个元素都会取到。

例如：

```python
for i in range(8):	#可以不写star和step，但结束位置一定要写的
    print(i)
print("---------")
for i in range(10, 2, -2):
    print(i)
```

输出结果：

```shell
0
1
2
3
4
5
6
7
---------
10
8
6
4
```

通过第一个 for 循环可以看出，range()的第一个元素的下标是从 `0` 开始，而不是从 1 开始；`range()`可以不写开始下标和步长，但一定得有结束位置；第二个 for 循环可以看出步长可以为负数，用于递减。

### continue 与 break

`continue` 的作用是跳过本次循环，后面的循环继续执行，例如：

```python
for i in range(1, 10):
    if i == 5:
        continue
    print(i)
```

输出结果：

```shell
1
2
3
4
6
7
8
9
```

很明显，`i`等于`5`的时候，for 循环就跳过去了，本次不再执行里面的代码，重新回到了新的循环。

同样的，还有终止所有循环的功能，就是 `break`，和 `continue` 是一样的用法，但效果是直接跳出整个循环。

例如：

```python
for i in range(1, 10):
    if i == 5:
        break
    print(i)
```

输出结果：

```shell
1
2
3
4
```

## 复杂数据结构

### 列表

列表是可以存放任何数据，包括整型，浮点型，字符串，布尔型等等，是常用的数据类型之一。

例如：

```python
# 列表的常用操作
# 创建列表
l = [1, 2, 3, 4, 5]  # 也可以是混合列表： l = [1,2.5,"a",True]
print(l)

# 访问列表元素
print(l[0])  # 1
print(l[-1])  # 5

# 列表长度
print(len(l))  # 5

# 追加元素
l.append(6)
print(l)

# 插入元素
l.insert(2, 7)
print(l)

# 删除元素
l.remove(7)
print(l)

# 切片
print(l[1:3])  # [2, 7]

# 列表排序
l.sort()
print(l)

# 反转列表
l.reverse()
print(l)

# 列表元素类型
print(type(l[0]))  # <class 'int'>
```

输出结果：

```shell
[1, 2, 3, 4, 5]
1
5
5
[1, 2, 7, 3, 4, 5, 6]
[1, 2, 3, 4, 5, 6]
[1, 2, 3, 4, 5, 6]
[2, 3, 4, 5, 6]
[6, 5, 4, 3, 2, 1]
<class 'int'>
```

### 元组

元组和列表类似，但元组是不可变的，即一旦创建就不能修改。

```python
# 元组的常用操作
# 创建元组
t = (1, 2, 3, 4, 5)
print(t)

# 访问元组元素
print(t[0])  # 1
print(t[-1])  # 5

# 元组长度
print(len(t))  # 5

# 切片
print(t[1:3])  # (2, 3)

# 元组元素类型
print(type(t[0]))  # <class 'int'>
```

输出结果：

```shell
(1, 2, 3, 4, 5)
1
5
5
(2, 3)
<class 'int'>
```

### 字典

字典是一种映射类型，它是无序的键值对集合。字典的每个键值对用冒号分割，键和值用逗号分割。

```python
# 字典的常用操作
# 创建字典
d = {"name": "张三", "age": 20, "gender": "男"}
print(d)

# 访问字典元素
print(d["name"])  # 张三
print(d.get("gender"))  # 男

# 字典长度
print(len(d))  # 3


# 字典键值对操作
d["city"] = "北京"
print(d)

d.pop("age")
print(d)

# 字典元素类型
print(type(d["name"]))  # <class'str'>
```

输出结果：

```shell
{'name': '张三', 'age': 20, 'gender': '男'}
张三
男
3
{'name': '张三', 'age': 20, 'gender': '男', 'city': '北京'}
{'name': '张三', 'gender': '男', 'city': '北京'}
<class'str'>
```

### 集合

集合是一种无序的元素集合，集合中的元素不能重复。

```python
# 集合的常用操作
# 创建集合
s = set([1, 2, 3, 4, 5])
print(s)

# 访问集合元素
print(s[0])  # 1
print(s[-1])  # 5

# 集合长度
print(len(s))  # 5

# 集合元素类型
print(type(s[0]))  # <class 'int'>

# 集合操作
s.add(6)
print(s)

s.remove(6)
print(s)

s1 = set([1, 2, 3])
s2 = set([2, 3, 4])
s3 = s1.union(s2)
print(s3)  # {1, 2, 3, 4}

s4 = s1.intersection(s2)
print(s4)  # {2, 3}

s5 = s1.difference(s2)
print(s5)  # {1}
```

输出结果：

```shell
{1, 2, 3, 4, 5}
1
5
5
<class 'int'>
{1, 2, 3, 4, 5, 6}
{1, 2, 3, 4, 5}
{1, 2, 3, 4}
{1}
```

### 字符串

在 Python 中，字符和字符串没有区别。

```python
# 字符串的常用操作
# 创建字符串
s = "hello world"
print(s)

# 访问字符串元素
print(s[0])  # h
print(s[-1])  # d

# 字符串长度
print(len(s))  # 11

# 字符串切片
print(s[1:5])  # ello

# 字符串元素类型
print(type(s[0]))  # <class'str'>

# 字符串操作
s1 = "hello"
s2 = "world"
s3 = s1 + " " + s2
print(s3)  # hello world

s4 = s1 * 3
print(s4)  # hellohellohello

s5 = "hello world"[6]
print(s5)  # o

s6 = "hello world".split()
print(s6)  # ['hello', 'world']

s7 = "hello world".replace("l", "x")
print(s7)  # hexxo worxd

s8 = "hello world".upper()
print(s8)  # HELLO WORLD

# 字符串格式化
name = "张三"
age = 20
print("我的名字是{}，今年{}岁。".format(name, age))  # 我的名字是张三，今年20岁。
```

输出结果：

```shell
hello world
h
d
11
ello
<class'str'>
hello world
hellohellohello
o
['hello', 'world']
hexxo worxd
HELLO WORLD
我的名字是张三，今年20岁。
```

## 函数

函数是由一组代码组成，完成某个特定的功能。

使用函数可以:

1. 避免代码的冗余
2. 提高代码的可维护性
3. 提高代码的可重用性
4. 提高代码的灵活性

### 定义函数

创建函数的语法如下：

```python
def 函数名(参数):
	代码块（函数的实现/函数体）
```

参数相当于变量,参数可以为 1 个或者多个，用逗号隔开，还可以没有参数，等于无参；代码块是函数的实现，又叫函数体。

### 函数的运行机制

函数的运行遵循以下机制：

1. 从函数调用开始执行
2. 通过函数名字找到函数定义的位置（创建函数的位置）
3. 执行函数体
4. 执行完毕之后，返回到函数的调用处

### 函数的使用

直接使用函数名调用即可，例如：

```python
# 定义函数
def say_hello():
    print("hello world")


# 调用函数
say_hello()


# 定义带参数的函数
def say_hello_with_name(name):
    print("hello " + name)


# 调用带参数的函数
say_hello_with_name("张三")
```

运行结果：

```shell
hello world
hello 张三
```

### 函数的参数

函数的参数首先要明白以下三个概念：

1. **形式参数（形参）**：在定义函数的时候传递的参数
2. **实际参数（实参）**：在调用函数时传递的参数
3. **无参**：没有任何参数

参数的使用：

- **位置参数**：实参的位置和形参一一对应，不能多也不能少。
- **关键字参数**：用形参的名字作为关键字来指定具体传递的值，则在函数调用时，前后顺序将不受影响。
- **位置参数和关键字参数混用**：当位置参数和关键字参数混用时，位置参数在前
- **默认参数**：给了默认值的参数--形参；如果传递了新的值，那么默认值就被覆盖了
- **可变成参数**：`def 函数名(*a)`本质上封装成了元组

例如：

```Python
# 定义函数
def add(a, b):
    return a + b


# 调用函数
print(add(1, 2))  # 3
print(add(2, 3))  # 5
print(add(a=1, b=2))  # 3
print(add(b=2, a=1))  # 3
print(add(1, 2, 3))  # 6
print(add(1, 2, 3, 4))  # 10
print(add(*[1, 2, 3, 4]))  # 10
```

输出结果：

```shell
3
5
3
3
6
10
10
```

### 函数的返回值

函数的返回值遵循以下规则：

1. 任何函数都有返回值
2. 如果不写 return ，也会默认加一个 return None
3. 如果写 return ，不写返回值 也会默认加一个 None
4. 可以返回任何数据类型
5. return 后面的代码不在执行，代表着函数的结束

例如：

```python
# 定义函数
def add(a, b):
    return a + b


# 调用函数
result = add(1, 2)
print(result)  # 3
```

输出结果：

```shell
3
```

### 函数文档

写代码的时候我们经常需要写文档，前面有提过`#`和三引号可以进行代码的注释，但在这里要介绍一种新的方法，也是写代码时常用的函数文档书写格式，这是标准化的函数文档书写：

```python
def 函数名(参数):
    """
    函数的描述信息
    参数:
        参数1: 参数1的描述信息
        参数2: 参数2的描述信息
    返回值: 函数的返回值描述信息
    """
    函数体
```

其中，函数的描述信息是必不可少的，参数的描述信息是可选的，返回值描述信息也是可选的。

### 作用域、内嵌函数与闭包

首先需要明白两个概念：局部变量和全局变量。

1. 局部变量：函数内部定义的变量，只能在函数内部访问，函数外部不能访问。
2. 全局变量：函数外部定义的变量，可以在整个程序范围内访问。

函数的变量作用域遵循以下规则：

1. 局部变量：函数内部定义的变量，只能在函数内部访问，函数外部不能访问。
2. 全局变量：函数外部定义的变量，可以在整个程序范围内访问。
3. 内置变量：在函数内部定义的变量，但不属于任何函数的变量，例如：`a=1`
4. 闭包变量：函数内部定义的变量，但不属于任何函数的变量，但是函数内部又使用了外部变量，这种变量称为闭包变量。

可以使用关键字来声明变量的作用域：

- `global`：声明全局变量
- `nonlocal`：声明闭包变量

例如：

```python
# 全局变量
a = 1


def func():
    # 局部变量
    b = 2
    # 闭包变量
    c = 3

    def inner_func():
        nonlocal c
        c = 4
        print(c)

    print(c)
    return inner_func


func()
func()()
# print(c)  # 错误，c 不是全局变量
```

输出结果：

```shell
3
4
```

### lambda 表达式

lambda 表达式是一种匿名函数，可以用来创建小型的函数。

语法：

```python
lambda 参数: 表达式
```

例如：

```python
# 匿名函数
add = lambda a, b: a + b
print(add(1, 2))  # 3
```

输出结果：

```shell
3
```

### 装饰器

装饰器是一种函数，它可以用来修改另一个函数的行为。

语法：

```python
@装饰器
def 函数名():
    函数体
```

例如：

```python
# 定义装饰器
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("before")
        result = func(*args, **kwargs)
        print("after")
        return result

    return wrapper


# 定义被装饰的函数
@my_decorator
def say_hello():
    print("hello world")


# 调用被装饰的函数
say_hello()
```

输出结果：

```shell
before
hello world
after
```

## 异常处理

1. 异常处理机制：当程序运行过程中出现异常时，Python 会自动生成一个异常对象，并将其抛出，程序员需要处理这个异常对象。
2. 异常处理的目的：通过异常处理机制，可以让程序在运行过程中更加健壮，更加安全。
3. 异常处理的原则：
   - 捕获异常：使用 try...except...finally 语句捕获异常。
   - 抛出异常：使用 raise 语句抛出异常。

语法：

```python

try:
    # 可能发生异常的代码
except 异常类型1:
    # 异常类型1的处理代码
except 异常类型2:
    # 异常类型2的处理代码
else:
    # 没有异常发生时的处理代码
finally:
    # 无论是否发生异常，都会执行的代码
```

其中：

1. `try` 代码块：可能发生异常的代码。
2. `except` 子句：异常类型及其对应的处理代码。
3. `else` 子句：没有异常发生时的处理代码。
4. `finally` 子句：无论是否发生异常，都会执行的代码。

例如：

```python
try:
    a = 1 / 0
except ZeroDivisionError:
    print("除数不能为 0")
else:
    print("没有异常发生")
finally:
    print("程序结束")
```

输出结果：

```shell
除数不能为 0
程序结束
```

## 结语

以上就是 Python 的语法基础，希望对你有所帮助！
