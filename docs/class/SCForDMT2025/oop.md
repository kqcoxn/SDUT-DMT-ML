# 从 C# 到面向对象

## 面向对象的概念和原理

### 什么是面向对象编程？

面向对象编程（Object-Oriented Programming，OOP）是一种常用的编程思想，它强调万物皆对象，因此在编程时我们可以将现实世界中的事物抽象成程序中的对象，从而更好实现软件的设计与开发。与传统的基于函数的编程不同，面向对象编程注重于将数据与行为封装在一起，即对象既包含数据状态，还包含可调用的行为方法。

面向对象编程的特点在于，它具有封装、继承和多态三大特性。封装意味着将对象的状态和行为进行封装，使其对外只暴露必要的接口，从而提高了安全性和可维护性；继承指的是某个对象可以继承另一个对象的特性，从而快速构建具有相似属性的对象；多态是指同一种行为在不同的对象上具有不同的表现形式，即在不同的情境下，同一个方法可以被不同的对象进行调用。

总之，面向对象编程是一种强大的编程方式，它具有高度封装性、灵活的继承性和强大的多态性，通过使用对象作为程序的基本处理单元，实现了数据和行为的有机结合，可以使程序更加高效、结构清晰，并方便管理和扩展。

### 面向对象的特征和优点

#### 封装

封装是 OOP 中最基本的特征之一，它将数据和方法封装在一个单独的单元中。对于实现封装，可以使用类来描述一个对象，类包括数据成员和成员函数。在类的定义中，可以使用关键字`public`、`protected`和`private`来指定成员访问权限，以保护数据的安全性。

```c#
public class Student
{
    // 公共成员变量
    public string Name { get; set; }

    // 私有成员变量
    private int _age;

    // 构造函数
    public Student(string name, int age)
    {
        Name = name;
        _age = age;
    }

    // 信息输出方法
    public void Info()
    {
        Console.WriteLine("姓名：" + Name);
        Console.WriteLine("年龄：" + _age);
    }
}
```

#### 继承

继承可以让子类继承父类的属性和方法，并且在基础上进行扩展。继承是代码复用的一种重要方式，它可以减少代码冗余，增加程序的可维护性。

```c#
class Animal
{
    protected string Name;

    public Animal(string name)
    {
        Name = name;
    }

    public void Eat()
    {
        Console.WriteLine(Name + "开始偷吃零食");
    }
}

class Cat : Animal
{
    public Cat(string name) : base(name) { }

    public void Run()
    {
        Console.WriteLine(Name + "nigirundayo!");
    }
}
```

#### 多态

多态指的是同一个行为在不同情况下有不同的表现形式。在 OOP 中，多态是一种通过继承、重写和接口实现的机制，它可以让不同类的对象对同一消息做出不同的响应。

```c#
abstract class Animal
{
    // 私有字段
    private string name;

    // 构造函数
    public Animal(string name)
    {
        this.name = name;
    }

    // 抽象方法
    public abstract void MakeSound();

    // 受保护的访问方法，供子类访问 name
    protected string GetName()
    {
        return name;
    }
}

class Dog : Animal
{
    public Dog(string name) : base(name) { }

    public override void MakeSound()
    {
        Console.WriteLine(GetName() + "狗叫！");
    }
}

class Cat : Animal
{
    public Cat(string name) : base(name) { }

    public override void MakeSound()
    {
        Console.WriteLine(GetName() + "哈气");
    }
}
```

## Python 中的面向对象编程基础

### 创建类和对象

首先，类是我们在面向对象编程中的基础，它是一种用来描述具有相同属性和方法的对象集合的蓝图。举个例子，我们可以创建一个名为 "人" 的类，这个类里面包含了姓名、年龄、性别等属性，以及 eat、sleep、work 等方法。然后我们可以实例化这个 "人" 类，创建很多具有不同属性的人的对象。

::: code-group

```Python
# 首先，我们来创建一个“人”类
class Person:
    # 把属性放在 __init__ 方法里面，注意第一个参数永远是 self，代表该类的实例
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender

    # 这里是一个日常生活中的行为，写成方法
    def eat(self):
        print("{} 在吃饭".format(self.name))

    # 再来一个睡觉
    def sleep(self):
        print("{} 正在睡觉".format(self.name))

    # 最后一个工作
    def work(self):
        print("{} 在工作".format(self.name))


# 现在我们创建两个人的对象
p1 = Person("小明", 18, "男")
p2 = Person("小红", 22, "女")

# 通过对象调用方法
p1.eat()  # 小明 在吃饭
p2.sleep()  # 小红 正在睡觉
```

```C#
class Person
{
    // 属性
    public string Name;
    public int Age;
    public string Gender;

    // 构造函数
    public Person(string name, int age, string gender)
    {
        Name = name;
        Age = age;
        Gender = gender;
    }

    // 方法：吃饭
    public void Eat()
    {
        Console.WriteLine($"{Name} 在吃饭");
    }

    // 方法：睡觉
    public void Sleep()
    {
        Console.WriteLine($"{Name} 正在睡觉");
    }

    // 方法：工作
    public void Work()
    {
        Console.WriteLine($"{Name} 在工作");
    }
}

// 主程序入口
class Program
{
    static void Main()
    {
        // 创建对象
        Person p1 = new Person("小明", 18, "男");
        Person p2 = new Person("小红", 22, "女");

        // 调用方法
        p1.Eat();   // 小明 在吃饭
        p2.Sleep(); // 小红 正在睡觉
    }
}
```

:::

### 初始化方法和实例属性

初始化方法是在创建一个新的对象实例时，执行其中的代码，以初始化对象的属性和状态。Python 中的初始化方法通常为 init()，它是所有类中的可选方法。当您创建一个类的新实例时，Python 将自动调用 init() 方法，并将该实例作为第一个参数传递给它，并使用该实例来设置各种属性和状态。

首先，我们定义了一个名为 Person 的类，并在其中编写了初始化方法 init()：

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

在上面的代码中，我们定义了一个新的类 Person，并在其中定义了初始化方法 init()。此方法将 name 和 age 作为参数传递，并使用 self.name 和 self.age 对象属性分别赋值。

接下来创建一个 Person 实例，并访问其中的属性：

```python
person1 = Person("小明", 20)
print("这个人的名字是：", person1.name) # 输出：这个人的名字是：小明
print("这个人的年龄是：", person1.age) # 输出：这个人的年龄是：20
```

在 Python 中，实例属性是方法内部定义的属性，它们与类的实例相关联，并且每个实例都有自己的一套实例属性。

### 类方法和静态方法的应用

在 Python 中，类方法和静态方法都是用来处理类的一些数据和行为的。两种方法都是通过类名进行调用，而不是通过实例对象进行调用。

#### 类方法

在 Python 中，定义类方法需要使用‘@classmethod’装饰器。类方法的第一个参数必须是‘cls’，表示类本身，可以通过‘cls’来调用类属性和类方法。

::: code-group

```Python
class Person:
    total_persons = 0

    def __init__(self, name):
        self.name = name
        Person.total_persons += 1

    @classmethod
    def show_total_persons(cls):
        print("Total persons: ", cls.total_persons)

p1 = Person("小祥")
p2 = Person("小睦")
Person.show_total_persons()
# 输出：Total persons:  2
```

```C#
class Person
{
    private static int totalPersons = 0;
    public string Name { get; }

    public Person(string name)
    {
        Name = name;
        totalPersons++;
    }

    public static void ShowTotalPersons()
    {
        Console.WriteLine("Total persons: " + totalPersons);
    }
}

class Program
{
    static void Main()
    {
        Person p1 = new Person("小祥");
        Person p2 = new Person("小睦");
        Person.ShowTotalPersons();
    }
}
```

:::

#### 静态方法

在 Python 中，定义静态方法需要使用‘@staticmethod’装饰器。静态方法没有参数限制，它只是一个普通函数，涉及到类和对象的问题，都需要在函数内部进行处理。

::: code-group

```Python
class Calculator:
    @staticmethod
    def add(x, y):
        return x + y

Calculator.add(3, 5) # 8
```

```C#
public class Calculator
{
    public static int Add(int x, int y)
    {
        return x + y;
    }
}

int result = Calculator.Add(3, 5);
```

:::

### 继承和子类化

首先，什么是继承？简单来说，它是一种让一个类从另一个类上继承它的属性和方法的方式。这个被继承的类称作“父类”，而这个继承的类则称作“子类”。例如，一个动物类可以是一个父类，而狗类和猫类可以是子类。

::: code-group

```Python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print("我是一只动物，我的名字是", self.name)

class Dog(Animal):
    def __init__(self, name):
        super().__init__(name)

    def bark(self):
        print("汪汪！")

class Cat(Animal):
    def __init__(self, name):
        super().__init__(name)

    def meow(self):
        print("哈气")


dog = Dog("旺财")
dog.speak() # 输出：我是一只动物，我的名字是 旺财
dog.bark()  # 输出：汪汪！

cat = Cat("耄耋")
cat.speak() # 输出：我是一只动物，我的名字是 耄耋
cat.meow()  # 输出：哈气
```

```C#
class Animal
{
    protected string name;

    public Animal(string name)
    {
        this.name = name;
    }

    public virtual void Speak()
    {
        Console.WriteLine("我是一只动物，我的名字是 " + name);
    }
}

class Dog : Animal
{
    public Dog(string name) : base(name) { }

    public void Bark()
    {
        Console.WriteLine("汪汪！");
    }
}

class Cat : Animal
{
    public Cat(string name) : base(name) { }

    public void Meow()
    {
        Console.WriteLine("哈气");
    }
}

class Program
{
    static void Main()
    {
        Dog dog = new Dog("旺财");
        dog.Speak();
        dog.Bark();

        Cat cat = new Cat("耄耋");
        cat.Speak();
        cat.Meow();
    }
}
```

:::

在两个子类中，我们通过调用 `super().__init__(name)` 方法让 Dog 和 Cat 的 name 属性继承自 Animal 的 name 属性。同时，我们也为它们分别添加了 bark 和 meow 方法，用于输出不同的声音。

我们通过继承让 Dog 和 Cat 类获得了 Animal 类的 speak 方法，并且通过子类化添加了不同的方法来实现功能的扩展。这样的方式既方便又灵活，可以大大简化我们的代码。

当然，这样的继承也可以产生一些问题，比如说实例化的对象可能会因为多层继承而产生命名冲突的问题。但是，在合理使用的情况下，它是一种非常有用的编程方式。

### 多重继承

多重继承是面向对象中非常重要且使用广泛的一种继承方式。它允许我们从多个父类中继承属性和方法，从而实现更加灵活的代码设计。但是，多重继承也可能会导致一些问题，例如方法名冲突、类的复杂性增加等等。因此，在使用多重继承时需要谨慎设计，尽量避免出现问题。

假设我们有 3 个类，分别是 Animal（动物）、Bird（鸟）和 Fish（鱼）。Animal 类是基类，它包含了所有动物都具有的属性和方法，例如 eat、sleep 等等。Bird 和 Fish 类都是继承自 Animal 类的子类，并且它们分别添加了自己独有的属性和方法，例如 Bird 类具有 fly 方法，而 Fish 类具有 swim 方法。

那么，如果我们需要创建一个 Penguin（企鹅）类，它既需要继承 Bird 类的 fly 方法，又需要继承 Fish 类的 swim 方法，该怎么办呢？这时候，就需要使用多重继承了。

::: code-group

```Python
class Animal:
    def __init__(self, name):
        self.name = name

    def eat(self):
        print(self.name + "正在吃东西...")

    def sleep(self):
        print(self.name + "正在睡觉...")

class Bird(Animal):
    def fly(self):
        print(self.name + "正在飞翔...")

class Fish(Animal):
    def swim(self):
        print(self.name + "正在游泳...")

class Penguin(Bird, Fish):
    def __init__(self, name):
        super().__init__(name)

    def run(self):
        print(self.name + "正在奔跑...")

penguin = Penguin("小企鹅")
penguin.fly()   # 输出：小企鹅正在飞翔...
penguin.swim()  # 输出：小企鹅正在游泳...
penguin.eat()   # 输出：小企鹅正在吃东西...
penguin.sleep() # 输出：小企鹅正在睡觉...
penguin.run()   # 输出：小企鹅正在奔跑...
```

```C#
class Animal
{
    protected string name;

    public Animal(string name)
    {
        this.name = name;
    }

    public void Eat()
    {
        Console.WriteLine(name + "正在吃东西...");
    }

    public void Sleep()
    {
        Console.WriteLine(name + "正在睡觉...");
    }
}

interface IBird
{
    void Fly();
}

interface IFish
{
    void Swim();
}

class Bird : Animal, IBird
{
    public Bird(string name) : base(name) { }

    public virtual void Fly()
    {
        Console.WriteLine(name + "正在飞翔...");
    }
}

class Fish : Animal, IFish
{
    public Fish(string name) : base(name) { }

    public virtual void Swim()
    {
        Console.WriteLine(name + "正在游泳...");
    }
}

class Penguin : Animal, IBird, IFish
{
    public Penguin(string name) : base(name) { }

    public void Fly()
    {
        Console.WriteLine(name + "正在飞翔...");
    }

    public void Swim()
    {
        Console.WriteLine(name + "正在游泳...");
    }

    public void Run()
    {
        Console.WriteLine(name + "正在奔跑...");
    }
}

class Program
{
    static void Main()
    {
        Penguin penguin = new Penguin("小企鹅");
        penguin.Fly();    // 小企鹅正在飞翔...
        penguin.Swim();   // 小企鹅正在游泳...
        penguin.Eat();    // 小企鹅正在吃东西...
        penguin.Sleep();  // 小企鹅正在睡觉...
        penguin.Run();    // 小企鹅正在奔跑...
    }
}
```

:::

在上面的代码中，我们定义了 Animal、Bird 和 Fish 三个类，它们分别继承自 Animal 类，并且添加了独有的属性和方法。接着，我们定义了 Penguin 类，它同时继承自 Bird 类和 Fish 类，并且添加了自己的 run 方法。这样，Penguin 类就可以同时拥有 Bird 类的 fly 方法和 Fish 类的 swim 方法了。
