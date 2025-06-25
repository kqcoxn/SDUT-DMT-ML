# Python 基础语法

## Python 环境安装与介绍

参考教程：[python 安装教程](https://blog.csdn.net/qq_44809707/article/details/122449191)、[pycharm 安装教程，超详细 - CSDN](https://blog.csdn.net/qq_44809707/article/details/122501118)

按照流程安装即可，注意路径最好不要出现中文。

## 基础操作

### 注释

在对应位置标记代码意义，增加可读性

```python
print('Hello World') # 输出“Hello World”
```

也可以使用多行注释，用三个双引号或者单引号括起来，可以将注释内容包裹起来

```python
"""
print('Hello World')
print('Hello World')
"""
print('Hello World') # 只会输出这一行
```

### 标准输出

`print()`函数：输出到控制台，可以输出多个值，用逗号隔开

```python
print('Hello World 1','Hello World 2','Hello World 3')
print()
print('Hello Python')
```

常用参数：

- `sep`：指定分隔符，默认是空格
- `end`：指定结尾符，默认是换行符
- `file`：指定输出文件，默认是控制台
- `flush`：立即输出，默认是 False

例如：

```python
import time

print("Hello", "world", sep=", ", end="!\n")

fp = open('test.txt', 'w') # 打开文件

# 使用print()函数写入文件
print('Hello, world!', file=fp)
time.sleep(2) # 等待2秒
print('Goodbye, world!', file=fp,flush=True)
time.sleep(2) # 等待2秒

fp.close() # 关闭文件
```

### 标准输入

`input()`

```python
input('请输入用户名：')
print({ok:1})
```

`input()`会等待用户输入，并返回输入的内容，可以用变量接收

```python
age = input('请输入年龄：')
print(age)
```

可以用`type()`函数判断输入的数据类型

```python
print(type(age))
```

### 变量

变量就是存储数据的容器，例如：求长方形面积，长 10，宽 5 变量 x、y、s

```python
x = 10
y = 5
s = x*y
print(s)
```

一个变量可以进行多次赋值，对应内存中改变的是指向

```python
x=100
x=50
print(x)
```

变量的名字是不固定的，我们可以自己定义，区分大小写。

在命名时需要注意：

- 不能以数字开头
- 不能包含空格、标点符号、特殊字符
- 尽量简短易懂
- 尽量避免与关键字冲突

例如：

```python
fast_food = True # 快餐
```

> 常见的关键字：`and`、`as`、`assert`、`break`、`class`、`continue`、`def`、`del`、`elif`、`else`、`except`、`False`、`finally`、`for`、`from`、`global`、`if`、`import`、`in`、`is`、`lambda`、`None`、`nonlocal`、`not`、`or`、`pass`、`raise`、`return`、`True`、`try`、`while`、`with`、`yield`

变量可以连续赋值，需要保证变量的个数和值的个数一致

```python
num1,num2=1,2

'''等同于：
num1=1
num2=2
'''

print(num1,num2)
```

交换两个变量的值（`swap()`函数）：

```python
num1=10
num2=20

temp=num1
num1=num2
num2=temp
'''等同于：
num1,num2=num2,num1
print(num1,num2)
'''

print(num1,num2)
```

### 数据类型

Python 中有五种基本数据类型：

- 整型（`int`）：整数，如 `1`、`2`、`3`
- 浮点型（`float`）：小数，如 `3.14`、`2.5`
- 字符串型（`str`）：字符串，如 `'Hello World'`、`''`
- 布尔型（`bool`）：布尔值，如 `True`、`False`
- 空值（`None`）：空值，如 `None`

Python 中还有一些复合数据类型：

- 列表（`list`）：有序集合，元素可以重复，如 `[1, 2, 3]`、`['apple', 'banana', 'orange']`
- 元组（`tuple`）：有序集合，元素不可修改，如 `(1, 2, 3)`、`('apple', 'banana', 'orange')`
- 字典（`dict`）：无序集合，元素是键值对，如 `{'name': 'Alice', 'age': 25}`
- 集合（`set`）：无序集合，元素不可重复，如 `{1, 2, 3}`、`{'apple', 'banana', 'orange'}`

> 注：分类方法可能与其他不同

数值类型也可以表示复数，由实部和虚部组成：

```python
com = complex(1,2)  # com=1+2i
print(com)
```

可以使用 `int()`、`float()`、`str()`、`bool()` 等函数进行类型转换：

```python
num = 123.1
print(num)

str_num = str(num)
print(str_num)

float_num = float(str_num)
print(float_num)

int_num = int(float_num)
print(int_num)

bool_num = bool(int_num) # 非0(0、0.0、’’、[]、{})即True
print(bool_num)
```

### 运算符

Python 中有以下运算符：

- 算术运算符：`+`、`-`、`*`、`/`、`//`、`%`，用于数值计算
- 赋值运算符：`=`、`+=`、`-=`、`*=`、`/=`、`//=`、`%=`，用于变量赋值
- 比较运算符：`==`、`!=`、`>`、`>=`、`<=`，用于比较两个值是否相等
- 逻辑运算符：`and`、`or`、`not`，用于逻辑判断
- 位运算符：`&`、`|`、`^`、`~`、`<<`、`>>`，用于按位操作
- 成员运算符：`in`、`not in`，用于判断元素是否存在于集合中
- 身份运算符：`is`、`is not`，用于判断两个变量是否指向同一个内存地址

运算符的优先级：

- 1. 圆括号 `()`
- 2. 乘除法 `*`, `/`, `//`, `%`
- 3. 加减法 `+`, `-`
- 4. 移位运算符 `<<`, `>>`
- 5. 位运算符 `&`, `^`, `|`
- 6. 比较运算符 `<`, `<=`, `>`, `>=`
- 7. 等于运算符 `==`, `!=`
- 8. 赋值运算符 `=`, `+=`, `-=`, `*=`, `/=`, `//=`, `%=`
- 9. 逻辑运算符 `not`, `and`, `or`
- 10. 成员运算符 `in`, `not in`
- 11. 身份运算符 `is`, `is not`

例如：

```python
a = 10
b = 5
b+=1
c = a + b * 2 / 3
print(c)

print(1>2 and 2>3)
print(1>2 and 3>2)
```

## 控制语句

### 分支控制语句

1.`if` 语句：

```python
if 条件表达式:
    语句块
```

2.`if-else` 语句：

```python
if 条件表达式:
    语句块1
else:
    语句块2
```

3.`if-elif-else` 语句：

```python
if 条件表达式1:
    语句块1
elif 条件表达式2:
    语句块2
elif 条件表达式3:
    语句块3
else:
    语句块4
```

例如，我有 200 元。我去买玩具，我需要对比玩具的价格和兜里的钱的大小：

```python
money = 10000
TV_money = int(input('请输入电视价格：'))
if 3000 <= TV_money < 6000:
    print('直接购买')
elif TV_money < 3000:
    print('太便宜，不买')
elif 6000 <= TV_money <= 10000:
    print('我再考虑考虑')
else:
    print('肯定不买')
```

> 注意：与 C 不同的是，Python 不需要使用`{}`来包裹语句块，而是使用缩进（Tab 键）来表示语句块的层次关系。

分支也可以嵌套：

```python
if 条件表达式1:
    符合条件表达式1执行的代码
    if 条件表达式1.1:
        符合条件表达式1并且符合条件表达式1.1执行的代码
    elif 条件表达式1.2:
        符合条件表达式1不符合条件表达式1.1并且符合条件表达式1.2执行的代码
elif 条件表达式2:
    不符合条件表达式1符合条件表达式2执行的代码
else:
    既不符合条件表达式1又不符合条件表达式2执行的代码
```

### 循环控制语句

循环控制语句可以分为 for 循环和 while 循环。

#### for 循环：

```python
for 临时变量 in 可迭代对象:
    循环体
```

可迭代对象包括`range()`函数、字符串、列表、字典等等

例如：

```python
for i in range(1, 10, 2):
    print(i)
```

其中，`range()`函数的三个参数分别是起始值、结束值、步长：

```python
range(start, stop, step)
```

- `start`：起始值，默认为 0
- `stop`：结束值，不包含该值
- `step`：步长，默认为 1

> 注意：`range()`函数至少要传一个参数，如果传一个参数，代表此参数是 `stop`

range()的参数可以为正，可以为负，步长的正负，可以理解成坐标轴的正负方向。

例如：

```python
for i in range(1,3,-1):
    print(i)
# 什么都不打印

for i in range(3,1,-1):
    print(i)
```

#### for-else 循环

- 如果循环正常退出(没有执行 `break`)，会执行 `else` 代码
- 如果循环不正常退出(有执行 `break`)，则不会执行 `else` 代码

控制循环的关键字：`break`（中止，强制退出循环） 和 `continue`（继续，退出本次循环）

- `break` 用于终止当前循环，并跳出循环体
- `continue` 用于跳过当前循环，直接进入下一次循环

例如：

```python
for i in range(1, 10):
    if i == 5:
        break
    print(i)
else:
    print('循环正常结束')

print('循环1结束')

for i in range(1, 10):
    if i == 5:
        continue
    print(i)
else:
    print('循环正常结束')
```

for 循环的循环嵌套：先运行外层循环，在运行内层循环

```python
for 临时变量1 in 可迭代对象:
    循环体
    for 临时变量2 in 可迭代对象:
        循环体
```

## 调试

1. 在需要断点的地方打上断点，点击左侧的小圆点，会出现一个红点
2. 点击右侧的调试按钮，会出现调试选项，选择 Python Debugging
3. 点击绿色的开始调试按钮，程序会暂停在断点处，可以查看变量的值、调用堆栈、控制台输出等
4. 点击红色的停止调试按钮，程序会继续运行

## 字符串

### 常用操作

只要被单引号或双引号或者三引号包裹起来的内容都是字符串。字符串属于不可变类型，字符串中的内容一旦被定义，则无法进行更改。

字符串不可于数字相加。

```python
name = 'zhangsan'
name = "zhangsan"
name = """zhangsan""" # 如果三引号没有变量接收，那么是注释，如果有变量接收，则为字符串
print(name)
```

字符串是一个有序的序列，可以用索引来访问字符串中的元素。索引从 0 开始，从左到右，从上到下。

```python
str = 'hello world'
print(str[0]) # h
print(str[6]) # d
print(str[-1]) # d
print(str[-2]) # l
```

字符串的常用操作有：

1. 字符串的拼接：`+`
2. 字符串的重复：`*`
3. 字符串的切片：`[start:end:step]`

例如：

```python
str1 = 'hello'
str2 = 'world'
str3 = str1 + str2
print(str3) # helloworld


str4 = 'hello' * 3
print(str4) # hellohellohello

str5 = 'hello world'
print(str5[0:5]) # hello
print(str5[6:11]) # world
print(str5[::2]) # hlowrd
```

如果我们想要倒叙输出一个字符串（将“python”输出成“nohtyp”）：

```
word = 'python'
A.word[-1:0:-1]  # nohty
B.word[::-1]     # nohtyp
C.word[-1::-1]   # nohtyp
D.word[-1:-7:-1] # nohtyp
```

正确答案：`B` `C` `D`

如果我们想遍历字符串：

```python
for char in 'hello world':
    print(char)
```

### 常用方法

字符串的方法很多，但我们最常用的其实就那么几个：

- `len()`：获取字符串的长度
- `find(str)`：查找子串的位置，如果不存在，返回 -1
- `replace(old, new)`：替换子串
- `split(str)`：以指定字符串分割字符串，返回列表
- `join(list)`：以指定列表中的元素连接字符串，返回字符串
- `isdigit()`：判断字符串是否为数字
- `isalpha()`：判断字符串是否为字母
- `isalnum()`：判断字符串是否为字母或数字
- `lower()`：转换为小写
- `upper()`：转换为大写
- `capitalize()`：首字母大写
- `title()`：每个单词首字母大写
- `strip()`：去除两端空格
- `lstrip()`：去除左侧空格
- `rstrip()`：去除右侧空格
- `startswith(str)`：判断字符串是否以指定字符串开头
- `endswith(str)`：判断字符串是否以指定字符串结尾
- `center(width)`：将字符串居中，并在两侧填充指定字符
- `ljust(width)`：将字符串左对齐，并在右侧填充指定字符
- `rjust(width)`：将字符串右对齐，并在左侧填充指定字符

看着很多，但实际上用到某个需求的时候直接查就行。

例如：

```python
# 字符串切分
str = 'hello world'
print(len(str)) # 11
print(str.find('l')) # 2
print(str.replace('l', 'L')) # heLLo world
print(str.split('o')) # ['hell', ' w', 'rld']
print(' '.join(['hello', 'world'])) # hello world

# 字符串判断
str = '12345'
print(str.isdigit()) # True
print(str.isalpha()) # False
print(str.isalnum()) # True

# 字符串转换
str = 'HELLO WORLD'
print(str.lower()) # hello world
print(str.upper()) # HELLO WORLD
print(str.capitalize()) # Hello world
print(str.title()) # Hello World

# 去除空格
str = 'hello world '
print(str.strip()) # hello world
print(str.lstrip()) # hello world
print(str.rstrip()) # hello world

# 字符串判断
str = 'hello world'
print(str.startswith('he')) # True
print(str.endswith('ld')) # True

# 字符串对齐
str = 'hello'
print(str.center(10, '*')) # **hello**
print(str.ljust(10, '*')) # hello****
print(str.rjust(10, '*')) # ****hello
```

如果一个方法返回的结果正好是下一步方法所需要的对象，那么可以使用链式调用，例如：

```python
str = '  hello world  '
new_str = str.strip().replace('l', 'w').upper()
print(new_str)  # HEWWO WORWD
```

### 字符串格式化

如果我们需要将多个参数按照指定格式输出，可以使用字符串格式化：

```python
name = 'zhangsan'
age = 25
print('我的名字是%s，今年%d岁了' % (name, age))
```

其中`%`为格式化符号，在字符串中使用 `%s`、`%d`、`%f` 等来表示字符串、整数、浮点数

也可以使用`format()`方法，在字符串中使用 `{}` 来表示需要输出的参数：

```python
name = 'zhangsan'
age = 25
print('我的名字是{0}，今年{1}岁了'.format(name, age))
```

其中， `{0}`、`{1}` 等为参数的编号，在 `format()` 方法中使用 `{}` 来表示需要输出的参数。

## 列表

列表是 Python 中最常用的复合数据类型，它可以存储多个元素，元素可以是任意类型。

### 常用操作

列表的创建：

```python
list1 = [1, 2, 3, 4, 5]
list2 = ['apple', 'banana', 'orange']
list3 = [1, 'apple', 3.14, True]
print(list1)
print(list2)
print(list3)
```

索引与赋值：

```python
list1 = [1, 2, 3, 4, 5]
print(list1[0]) # 1
list1[0] = 10
print(list1) # [10, 2, 3, 4, 5]
```

列表支持序列解包操作：

```python
list1 = [1, 2, 3, 4, 5]
a, b, *c, d, e = list1
print(a) # 1
print(b) # 2
print(c) # [3, 4]
print(d) # 5
print(e) # 5
```

列表的切片操作与字符串的一样，但返回的是列表：

```python
list1 = [1, 2, 3, 4, 5]
print(list1[1:3]) # [2, 3]
print(list1[::2]) # [1, 3, 5]
```

### 列表方法

常用的列表方法有：

- `len()`：获取列表的长度
- `append(obj)`：在列表末尾添加元素
- `pop(index)`：删除指定位置的元素，并返回该元素
- `extend(list)`：在列表末尾添加另一个列表
- `index(obj)`：查找元素的位置，如果不存在，抛出 `ValueError` 异常
- `count(obj)`：统计元素出现的次数
- `sort()`：对列表进行排序
- `reverse()`：对列表进行反转
- `clear()`：清空列表
- `insert(index, obj)`：在指定位置插入元素
- `remove(obj)`：删除指定元素，如果不存在，抛出 `ValueError` 异常
- `copy()`：复制列表

例如：

```python
list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9, 10]
list1.append(6)
list1.pop(2)
list1.extend(list2)
print(list1) # [1, 2, 4, 5, 6, 7, 8, 9, 10]
print(list1.index(5)) # 3
print(list1.count(5)) # 1
list1.sort()
list1.reverse()
list1.clear()
print(list1) # []
list1.insert(0, 1)
list1.remove(1)
list1.append(2)
list1_copy = list1.copy()
print(list1_copy) # [2]
```

### 列表推导式

列表推导式是一种创建列表的简洁方式，可以根据某种条件筛选出符合条件的元素，并将其转换为新的列表。

语法：

```python
new_list = [expression for item in iterable if condition]
```

其中：

- `expression`：表达式，用于生成新元素
- `item`：可迭代对象中的元素
- `iterable`：可迭代对象，如列表、元组、字符串
- `condition`：可选，用于筛选元素的条件

例如：

```python
list1 = [1, 2, 3, 4, 5]
list2 = [x**2 for x in list1 if x % 2 == 0]
print(list2) # [4, 16]
```

### 列表嵌套

列表可以嵌套列表，即一个列表的元素可以是多个其他列表，一般用于表示多维数组。

例如：

```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]
list4 = [list1, list2, list3]
print(list4) # [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

## 字典

字典是 Python 中另一种常用的复合数据类型，它是无序的键值对集合，使用`{"key":"value"}`，键值对来确定字典，以逗号分隔，以大括号去包裹的序列。

### 常用操作

字典的创建：

```python
dict1 = {'name': 'zhangsan', 'age': 25}
dict2 = {'name': 'lisi', 'age': 26}
dict3 = {'name': 'wangwu', 'age': 27}
print(dict1)
print(dict2)
print(dict3)
```

索引与赋值：

```python
dict1 = {'name': 'zhangsan', 'age': 25}
print(dict1['name']) # zhangsan
dict1['age'] = 26
print(dict1) # {'name': 'zhangsan', 'age': 26}
```

字典的键必须是不可变类型，如字符串、数字、元组等。

### 字典方法

常用的字典方法有：

- `len()`：获取字典的长度
- `keys()`：获取字典的键列表
- `values()`：获取字典的值列表
- `items()`：获取字典的键值对列表
- `get(key, default=None)`：获取指定键的值，如果不存在，返回默认值
- `pop(key)`：删除指定键的值，并返回该值
- `popitem()`：随机删除一个键值对，并返回该键值对
- `update(dict)`：更新字典
- `clear()`：清空字典
- `dict()`：将序列转换为字典

例如：

```python
dict1 = {'name': 'zhangsan', 'age': 25}
dict2 = {'name': 'lisi', 'age': 26}
dict1.update(dict2)
print(dict1) # {'name': 'lisi', 'age': 26}
print(dict1.keys()) # dict_keys(['name', 'age'])
print(dict1.values()) # dict_values(['lisi', 26])
print(dict1.items()) # dict_items([('name', 'lisi'), ('age', 26)])
print(dict1.get('name')) # lisi
print(dict1.get('gender', 'unknown')) # unknown
dict1.pop('age')
print(dict1) # {'name': 'lisi'}
dict1.popitem()
print(dict1) # {}
dict1.clear()
print(dict1) # {}
dict3 = dict([('name', 'zhangsan'), ('age', 25)])
print(dict3) # {'name': 'zhangsan', 'age': 25}
```

## 元组

元组是 Python 中另一种不可变的复合数据类型，它是一系列不可变的元素组成的序列，使用`()`来表示。

### 常用操作

元组的创建：

```python
tuple1 = (1, 2, 3, 4, 5)
tuple2 = ('apple', 'banana', 'orange')
tuple3 = (1, 'apple', 3.14, True)
print(tuple1)
print(tuple2)
print(tuple3)
```

索引与赋值：

```python
tuple1 = (1, 2, 3, 4, 5)
print(tuple1[0]) # 1
tuple1[0] = 10 # TypeError: 'tuple' object does not support item assignment
```

元组的切片操作与字符串、列表的一样，但返回的是元组：

```python
tuple1 = (1, 2, 3, 4, 5)
print(tuple1[1:3]) # (2, 3)
print(tuple1[::2]) # (1, 3, 5)
```

### 常用方法

常用的元组方法有：

- `len()`：获取元组的长度
- `count(obj)`：统计元素出现的次数
- `index(obj)`：查找元素的位置，如果不存在，抛出 `ValueError` 异常
- `tuple()`：将序列转换为元组
- `map(func, iterable)`：将函数应用到序列的每个元素，并返回结果的元组
- `zip(iterable1, iterable2)`：将两个序列的元素打包成一个元组列表
- `del()`：删除元组

例如：

```python
tuple1 = (1, 2, 3, 4, 5)
print(len(tuple1)) # 5
print(tuple1.count(3)) # 1
print(tuple1.index(3)) # 2
tuple2 = tuple(range(1, 6))
print(tuple2) # (1, 2, 3, 4, 5)
tuple3 = tuple(map(lambda x: x**2, range(1, 6)))
print(tuple3) # (1, 4, 9, 16, 25)
tuple4 = tuple(zip('abc', range(1, 4)))
print(tuple4) # (('a', 1), ('b', 2), ('c', 3))
del(tuple1) # TypeError: 'tuple' object doesn't support item deletion
```

### 元组与列表的区别

1. 元组的元素不能修改，不能添加或删除元素，只能读取元素。
2. 元组的元素是不可变的，因此元组是不可变的，而列表是可变的。
3. 元组的创建速度比列表快，因为元组是不可变的，因此不需要创建新的对象，而列表是可变的，因此需要创建新的对象。
4. 元组的元素是有序的，列表的元素是无序的。

## 文件操作

### open()函数

内置函数`open()`可以打开文件，并返回一个文件对象，可以对文件进行读写操作，其接收参数如下：

```python
open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
```

其中：

- `file`：文件名或文件描述符，如果是文件名，则打开文件，如果是文件描述符，则直接使用该文件。
- `mode`：打开文件的模式，默认为`r`，表示只读模式，其他可选模式有：
  - `r`：只读模式，文件必须存在，否则抛出`FileNotFoundError`异常。
  - `w`：写入模式，文件不存在则创建，存在则覆盖，如果文件不存在，则创建文件，如果文件存在，则清空文件内容。
  - `x`：新建模式，文件必须不存在，否则抛出`FileExistsError`异常。
  - `a`：追加模式，文件不存在则创建，存在则在文件末尾追加。
  - `+`：读写模式，文件必须存在，否则抛出`FileNotFoundError`异常。
  - `U`：文本模式，文件必须存在，否则抛出`FileNotFoundError`异常。
- `buffering`：缓冲区大小，默认为`-1`，表示系统默认缓冲区大小。
- `encoding`：编码格式，默认为`None`，表示系统默认编码格式。
- `errors`：错误处理方案，默认为`None`，表示系统默认错误处理方案。
- `newline`：行结束符，默认为`None`，表示系统默认行结束符。
- `closefd`：是否关闭文件描述符，默认为`True`，表示关闭。
- `opener`：用于指定打开文件的函数，默认为`None`，表示系统默认打开函数。

一般我们只需要用到前三个参数。

当对文件进行读写操作时，需要先打开文件，然后使用文件对象的`read()`、`write()`、`readline()`等方法进行读写操作，最后需要使用`close()`方法关闭文件。

常用的方法包括：

- `read(size=-1)`：读取文件内容，如果`size`为`-1`，则读取整个文件，否则读取指定大小的内容。
- `readline(size=-1)`：读取一行内容，如果`size`为`-1`，则读取一整行，否则读取指定大小的内容。
- `readlines()`：读取所有行内容，并返回列表。
- `write(string)`：写入内容到文件末尾。
- `seek(offset, whence=0)`：移动文件读取指针到指定位置。
- `tell()`：获取文件读取指针当前位置。
- `close()`：关闭文件。

例如：

```python
# 打开文件
f = open('test.txt', 'w')

# 写入内容
f.write('hello world\n')
f.write('hello python\n')

# 读取内容
f.seek(0)
print(f.read())

# 关闭文件
f.close()
```

### with 语句

Python 2.5 引入了`with`语句，可以自动帮我们调用`open()`和`close()`方法，简化代码。

例如：

```python
with open('test.txt', 'w') as f:
    f.write('hello world\n')
    f.write('hello python\n')
    f.seek(0)
    print(f.read())
```

使用 with 语句，可以自动帮我们调用`open()`和`close()`方法，并保证文件正确关闭。

## os 模块

Python 的 os 模块提供了非常丰富的文件和目录操作函数，可以用来处理文件和目录。

- `os.name`：获取操作系统类型，`posix`表示 Linux、Unix、Mac OS X 等，`nt`表示 Windows。
- `os.getcwd()`：获取当前工作目录。
- `os.chdir(path)`：改变当前工作目录。
- `os.listdir(path)`：列出指定目录下的所有文件和目录。
- `os.mkdir(path)`：创建目录。
- `os.makedirs(path)`：递归创建目录。
- `os.remove(path)`：删除文件。
- `os.rename(src, dst)`：重命名文件或目录。
- `os.stat(path)`：获取文件或目录的状态信息。
- `os.path.join(path, *paths)`：拼接路径。
- `os.path.exists(path)`：判断文件或目录是否存在。
- `os.path.isfile(path)`：判断是否为文件。
- `os.path.isdir(path)`：判断是否为目录。
- `os.path.getsize(path)`：获取文件大小。
- `os.path.splitext(path)`：分离文件名与扩展名。

例如：

```python
import os

# 获取当前工作目录
print(os.getcwd())
```
