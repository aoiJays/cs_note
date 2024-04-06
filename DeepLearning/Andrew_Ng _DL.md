# 深度学习 - 吴恩达

> [Bilibili](https://www.bilibili.com/video/BV1FT4y1E74V)
>
> [配套资源](https://blog.csdn.net/weixin_36815313/article/details/105728919)

[TOC]

## 神经网络与深度学习

### 神经网络基础

对于结构化的训练数据，我们记列向量$x^{(i)}$所构成的矩阵为$X$

所有label值构成一个行向量$Y = \begin{bmatrix}  y^{(1)} & y^{(2)} & ... & y^{(m)}  \end{bmatrix}$

#### 梯度下降

懂的都懂
$$
w:= w - \alpha \frac{\partial J(w, b)  }{\partial w} \\
b:= b - \alpha \frac{\partial J(w, b)  }{\partial b}
$$

#### 计算图

令$J(a,b,c) = 3(a + bc)$

我们将此成本函数进行分解，引入中间变量
$$
u = bc\\
v = a+u\\
J = 3v
$$
![image-20240406183120119](./Andrew_Ng _DL.md.assets/image-20240406183120119.png)

- $\frac{\partial J}{\partial v} = 3$
- $\frac{\partial J}{\partial a} = \frac{\partial J}{\partial v}\times \frac{\partial v}{\partial a} = 3 \times 1 = 3$
- $\frac{\partial J}{\partial u} = \frac{\partial J}{\partial v}\times\frac{\partial v}{\partial u} = 3\times 1 = 3$
- $\frac{\partial J}{\partial b} = \frac{\partial J}{\partial u}\times\frac{\partial u}{\partial b}  = 3c $
- $\frac{\partial J}{\partial c} = \frac{\partial J}{\partial u}\times\frac{\partial u}{\partial c}  = 3b $



通过构建计算图，我们从右往左，可以非常容易计算出导数

每个矩形框使用的运算类型，其所代表的求导公式可以提前保存

#### Logistic Regression

> 给定$X$，需要计算出$\hat y = P(y=1|x)$
>
> 即符合某种分类的概率

- 输入：$X$
- 参数：$w\in R^{n}, b \in R$
- 输出：$\hat y = \sigma(w^Tx + b)$

> sigmoid函数：
> $$
> \sigma(z) = \frac{1}{1 + e^{-z}}
> $$
> 求导$y = \sigma(z)$：
> $$
> y = \frac{1}{1 + e^{-z}} \\
> y' = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1+e^{-x}- 1}{1+e^{-x}} \times \frac{1}{1+e^{-x}}=(1-y)y
> $$
> 

![image-20240406170939098](./Andrew_Ng _DL.md.assets/image-20240406170939098.png)

##### Loss Function损失函数

$$
L(\hat y, y) = -(y\log \hat y + (1-y)\log (1-\hat y))
$$

Loss Function是对单个样本的计算

##### Cost Function成本函数 

$$
J(w, b) = \frac{1}{m}L(\hat y^{(i)},y^{(i)})
$$

针对总体成本的函数

神经网络训练目标即为：找到最佳的参数，使得成本函数最小



##### Logistics Regression的梯度下降

首先考虑**单个样本**
$$
z^{(i)} = w^Tx^{(i)} + b \\
\hat y^{(i)} = \sigma(z^{(i)}) \\
L = -( y^{(i)}\log \hat y^{(i)} +(1- y^{(i)})\log  (1-\hat y^{(i)}) )
$$
我们求出梯度，通常在编程中，我们喜欢使用$d_a$表示目标函数关于$a$的梯度
$$
d\hat y^{(i)} = \frac{\partial L}{\partial \hat y^{(i)}} = - \frac{y^{(i)}}{\hat y^{(i)}} +\frac{1 - y^{(i)}}{1 - \hat y^{(i)}} \\
dz^{(i)} = d\hat y^{(i)} \times \frac{\partial \hat y^{(i)}}{\partial z^{(i)}} = d\hat y^{(i)} \times (1-\hat y^{(i)})\hat y^{(i)} = \hat y^{(i)} - y^{(i)}
$$

$w$我们**暂时先不看作是一个矩阵向量**

我们分别对$w_1,w_2, ...$即$w_j$进行求导
$$
dw_j = dz^{(i)}\times \frac{\partial z^{(i)}}{\partial w_j} = x^{(i)}_jdz^{(i)} =  x^{(i)}_j(\hat y^{(i)} - y^{(i)}) \\
d_b = dz^{(i)}\times \frac{\partial z^{(i)}}{\partial b} = dz^{(i)} = \hat y^{(i)} - y^{(i)}
$$
现在我们考虑**多个样本**，即需要考虑的是成本函数$J$​

事实上：$J = \frac{1}{m}L(\hat y^{(i)}, y^{(i)})$

每个样本之间都是不相干的，我们一个枚举，把所有单个样本的结果：$d_w,d_b$全部计算出来求和，最后除以$m$求平均即可

![image-20240406232344610](./Andrew_Ng _DL.md.assets/image-20240406232344610.png)

但是对于深度学习问题，我们枚举计算非常低效

解决方法有：

- 向量化
- 小批量随机梯度下降

