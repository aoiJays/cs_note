# 深度学习 - 吴恩达

> [Bilibili](https://www.bilibili.com/video/BV1FT4y1E74V)
>
> [配套资源](https://blog.csdn.net/weixin_36815313/article/details/105728919)

[TOC]

## 神经网络与深度学习

### 神经网络基础

对于结构化的训练数据，我们记列向量$x^{(i)}$所构成的矩阵为$X$

所有label值构成一个行向量$Y = \begin{bmatrix}  y^{(1)} & y^{(2)} & ... & y^{(m)}  \end{bmatrix}$



#### 激活函数

对于神经元来说，总是需要一个激活函数的

每个神经元将所有输入进行线性组合后，需要喂入激活函数，输出一个值



-   Sigmoid函数：

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

![image-20240406170939098](./Andrew_Ng _DL.assets/image-20240406170939098.png)



-   tanh函数

>   基本已经不用sigmoid激活函数了，tanh函数在所有场合都优于sigmoid函数
>
>   除了二分类问题：需要输出0-1的概率值
>
>   tanh可以认为是sigmoid的平移伸缩版本
>   $$
>   tanh(z) = \frac{e^z-e^{-z}}{e^z+e^{-z}}
>   $$
>   求导：
>   $$
>   \frac{d}{dz}tanh(z) = \frac{(e^z+e^{-z})(e^z+e^{-z})-(e^z-e^{-z})(e^z-e^{-z})}{(e^z+e^{-z})^2} = 1 - \frac{(e^z-e^{-z})^2}{(e^z+e^{-z})^2} = 1-tanh^2(z)
>   $$
>   

![image-20240408194918786](./Andrew_Ng _DL.assets/image-20240408194918786.png)



-   Relu

>   $$
>   Relu(z) = max(0, z)
>   $$
>
>   在0处导数不存在，但是全部取0的概率非常低

![image-20240408195910390](./Andrew_Ng _DL.assets/image-20240408195910390.png)

-   Leaky Relu

$$
Relu(x) = max(0.01x, x)
$$

![image-20240408200058217](./Andrew_Ng _DL.assets/image-20240408200058217.png)



一般推荐使用Relu即可，训练速度快于前两者

**sigmoid**和**tanh**函数的导数在正负饱和区的梯度都会接近于0，这会造成梯度弥散

**Relu**进入负半区的时候，梯度为0，神经元此时不会训练，产生所谓的稀疏性，而**Leaky ReLu**不会有这问题

但有足够的隐藏层使得z值大于0，所以对大多数的训练数据来说学习过程仍然可以很快

>-   梯度弥散：反向传播中，导数连乘，若梯度很小（小于1），就会使得越远离输出层的梯度越小，越靠近输出层的梯度越大。靠近输入层的梯度趋近于0，基本不训练，无法学习输入层特征
>-   梯度爆炸：导数很大，数值溢出



##### 激活函数的意义

我们令激活函数$g(x)=x$，相当于什么都不用做

经过第一层神经网络：$a^{[1]} = W^{[1]}x+b^{[1]}$

经过第二层神经网络：$a^{[2]} = W^{[2]}a^{[1]}+b^{[2]}$

我们展开即得：
$$
a^{[2]} = W^{[2]}W^{[1]}x + (W^{[2]}b^{[1]}+b^{[2]})= Wx+b
$$
因此无论网络多么复杂，最后得到的结果都是$x$的线性组合

- 输出层可以使用$g(x)=x$线性激活函数（预测房价的结果）



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
![image-20240406183120119](./Andrew_Ng _DL.assets/image-20240406183120119.png)

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


##### Loss Function损失函数

$$
L(\hat y, y) = -(y\log \hat y + (1-y)\log (1-\hat y))
$$

Loss Function是对单个样本的计算

>**为什么选择上述函数作为损失函数？**
>
>我们希望我们的模型输出$\hat y$：标签$y=1$的概率
>
>因此，标签$y=0$的概率即为：$1-\hat y$
>
>即为：
>$$
>P(y=1|x) = \hat y\\
>P(y=0|x) = 1-\hat y
>$$
>当y=1时，我们期待$\hat y$尽可能大，接近1
>
>当y=0时，我们期待$1-\hat y$尽可能大，接近1
>
>
>
>我们希望统一这两个式子：
>$$
>P(y|x) = \hat y^y\times (1-\hat y)^{1-y}
>$$
>不符合的地方自动会变成1，不影响到结果
>
>**因此目标为：最大化这个函数**
>
>但这个函数比较抽象，不方便研究
>
>我们知道对数函数是一个**单调递增**的函数，因此我们对原式同时进行对数操作
>$$
>\log P(y|x)= \log( \hat y^y\times (1-\hat y)^{1-y})= y\log \hat y+(1-y)\log (1-\hat y)
>$$
>我们希望最大化$P(y|x)$，那么根据单调函数的性质，最大化$\log P(y|x)$即可
>
>我们需要最大化左边的值，就需要最大化右边
>
>我们加个负号
>$$
>L(\hat y, y) = -(y\log \hat y + (1-y)\log (1-\hat y))
>$$
>右边的表达式就是损失函数的负值
>
>我们最小化损失函数，与最大化概率值的目的一致
>
>因此作为损失函数非常不错




##### Cost Function成本函数 

$$
J(w, b) = \frac{1}{m}\sum L(\hat y^{(i)},y^{(i)})
$$

针对总体成本的函数

神经网络训练目标即为：找到最佳的参数，使得成本函数最小

>   **为什么选择这个作为成本函数？**
>
>   ~~直觉上是这样~~
>
>   我们考虑从**最大似然估计**的角度出发
>
>   在参数$w,b$下，所有单个样本等于某个随机值$y_i$的概率就是：$\prod  P(y_i|x)$
>
>   构造似然函数：$L(w,b|y,x) = \prod  P(y_i|x)$
>
>   是一个关于$w,b$的函数，$y,x$均为定值
>
>   同样，我们希望两边求对数
>   $$
>   \log L =- \sum  L(\hat y, y)
>   $$
>   我们令成本函数：
>   $$
>   J(w, b) = -\frac{1}{m} \sum  L(\hat y, y)
>   $$
>   乘上一个$\frac{1}{m}$常数因子没什么影响，更加方便
>
>   最小化成本函数，即为最大化似然函数，则就是找到了最佳参数组合，使得样本和标签值已知的情况下，概率最大




##### Logistics Regression的梯度下降

首先考虑**单个样本**
$$
z^{(i)} = w^Tx^{(i)} + b \\
\hat y^{(i)} = \sigma(z^{(i)}) \\
L = -( y^{(i)}\log \hat y^{(i)} +(1- y^{(i)})\log  (1-\hat y^{(i)}) )
$$
**我们求出梯度，通常在编程中，我们喜欢使用$d_a$表示目标函数关于$a$的梯度**
$$
d\hat y^{(i)} = \frac{\partial L}{\partial \hat y^{(i)}} = - \frac{y^{(i)}}{\hat y^{(i)}} +\frac{1 - y^{(i)}}{1 - \hat y^{(i)}} \\
dz^{(i)} = d\hat y^{(i)} \times \frac{\partial \hat y^{(i)}}{\partial z^{(i)}} = d\hat y^{(i)} \times (1-\hat y^{(i)})\hat y^{(i)} = \hat y^{(i)} - y^{(i)}
$$

$w$我们**暂时先不看作是一个矩阵向量**

我们分别对$w_1,w_2, ...$即$w_j$进行求导
$$
dw_j^{(i)} = dz^{(i)}\times \frac{\partial z^{(i)}}{\partial w_j} = x^{(i)}_jdz^{(i)} =  x^{(i)}_j(\hat y^{(i)} - y^{(i)}) \\
db^{(i)} = dz^{(i)}\times \frac{\partial z^{(i)}}{\partial b} = dz^{(i)} = \hat y^{(i)} - y^{(i)}
$$
现在我们考虑**多个样本**，即需要考虑的是成本函数$J$​

事实上：$J = \frac{1}{m}L(\hat y^{(i)}, y^{(i)})$

每个样本之间都是不相干的，我们一个枚举，把所有单个样本的结果：$d_w,d_b$全部计算出来求和，最后除以$m$求平均即可

![image-20240406232344610](./Andrew_Ng _DL.assets/image-20240406232344610.png)

但是对于深度学习问题，我们枚举计算非常低效

解决方法有：

- 向量化
- 小批量随机梯度下降

 

##### Logistics Regression的梯度下降（向量化版本）

>   避免显示的for循环
>
>   不管是CPU还是GPU，我们都更加推荐使用支持SIMD的函数
>
>   充分并行化操作，加快执行效率

-   $v = [v_1, ..., v_n]^T$
-   $e^v = [e^{v_1},...,e^{v_n}]^T$

大部分函数操作，如`abs,log`等，都是会自动扩展成矩阵向量形式的

---

我们首先计算$z^{(i)}$，非向量版本就需要一个个枚举
$$
z^{(i)} = w^Tx^{(i)} + b
$$
我们可以放在一个矩阵中
$$
z =\begin{bmatrix}
z^{(1)} & z^{(2)} & ... & z^{(m)} 
\end{bmatrix}
= 
\begin{bmatrix}
 w^Tx^{(1)} + b & w^Tx^{(2)} + b & ... &  w^Tx^{(m)} + b
\end{bmatrix}
=
w^TX+b
$$
则有：
$$
\hat y =\begin{bmatrix}
\hat y^{(1)} & \hat y^{(2)} & ... & \hat y^{(m)} 
\end{bmatrix}
=
\begin{bmatrix}
\sigma(z^{(1)}) & \sigma(z^{(2)}) & ... & \sigma(z^{(m)} )
\end{bmatrix}
=
\sigma\begin{bmatrix}
z^{(1)} & z^{(2)} & ... & z^{(m)} 
\end{bmatrix}
= \sigma(z)
$$
接下来我们计算梯度
$$
dz = \begin{bmatrix}
dz^{(1)} & dz^{(2)} & ... & dz^{(m)} 
\end{bmatrix}
=
\begin{bmatrix}
\hat y^{(1)}- y^{(1)} & \hat y^{(2)}-y^{(2)} & ... & \hat y^{(m)}-y^{(m)} 
\end{bmatrix}
= \hat y - Y
$$
**这也就解释了为什么Y是行向量**

>   列向量表示同一个样本的不同属性
>
>   行向量表示多个不同样本


$$
dw = \begin{bmatrix}
dw_1\\
dw_2\\
...\\
dw_n
\end{bmatrix}
=
\frac{1}{m} \begin{bmatrix}
\sum x_1^{(i)}dz^{(i)}  \\
\sum x_2^{(i)}dz^{(i)}\\
...\\
\sum x_n^{(i)}dz^{(i)}
\end{bmatrix}
=
\frac{1}{m} \begin{bmatrix}
x_1dz^T  \\
x_2dz^T \\
...\\
x_ndz^T 
\end{bmatrix}
=\frac{1}{m}Xdz^T
$$
得到的是一个$(n\times m)*(m\times 1) = (n\times 1)$的矩阵
$$
db = \frac{1}{m}\sum dz^{(i)}
$$
是一个标量

最后，梯度下降即可以表示为：
$$
w:=w -\alpha dw\\
b:=b -\alpha db
$$



### 浅层神经网络

![image-20240408165222042](./Andrew_Ng _DL.assets/image-20240408165222042.png)

- 一般忽略输入层，称为双层神经网络
- 这里我们使用中括号表示层编号

#### 正向传播：单个样本的计算

我们有一个需要计算的样本$x$，现在需要通过神经网络计算其$\hat y$

以上述双层神经网络为例，第一层的结果为：
$$
a^{[1]} = \begin{bmatrix}
a^{[1]}_1\\
a^{[1]}_2\\
a^{[1]}_3\\
a^{[1]}_4\\
\end{bmatrix}

= \sigma \begin{bmatrix}
w^{[1]T}_1x+b^{[1]}_1  \\
w^{[1]T}_2x+b^{[1]}_2\\
w^{[1]T}_3x+b^{[1]}_3\\
w^{[1]T}_4x+b^{[1]}_4\\
\end{bmatrix}
=\sigma(
\begin{bmatrix}
...w^{[1]}_1...\\
...w^{[1]}_2...\\
...w^{[1]}_3...\\
...w^{[1]}_4...\\
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2\\
x_3\\
\end{bmatrix} +
\begin{bmatrix}
b^{[1]}_1\\
b^{[1]}_2\\
b^{[1]}_3\\
b^{[1]}_4\\
\end{bmatrix}
)
$$
需要注意，$w^{[i]}_j$是每个神经元的参数向量，之前的表示是一个列向量

这里我们把它变成行向量，构成矩阵$W^{[1]}$的一个行向量

则有：
$$
a^{[1]} = \sigma(W^{[1]}x+b^{[1]})
$$
对于第二层也是同理进行计算即可
$$
a^{[2]} = \sigma(W^{[2]}a^{[1]}+b^{[2]})
$$


#### 正向传播：多个样本的计算

我们最终需要得到矩阵$A$​

首先计算第一层神经网络的结果：
$$
A^{[1]} = 
\begin{bmatrix}
a^{[1](1)} & a^{[1](2)} & ... &a^{[1](m)}\\
\end{bmatrix}
$$
每个列向量$a^{[1](i)}$表示**第$i$个样本**经过第一层神经网络计算得到的结果
$$
A^{[1]} &= 
\sigma\begin{bmatrix}
W^{[1]}x^{(1)}+b^{[1]}& W^{[1]}x^{(2)}+b^{[1]} & ... &W^{[1]}x^{(m)}+b^{[1]}\\
\end{bmatrix} \\
&= \sigma (W^{[1]} \begin{bmatrix} x^{(1)} & x^{(2)} & ... & x^{(m)}   \end{bmatrix} +b^{[1]}) \\
&=\sigma (W^{[1]} X   +b^{[1]})
$$
第二层同理：
$$
A^{[2]} = \sigma (W^{[1]} A^{[1]}   +b^{[1]})
$$

#### 反向传播：单个样本的计算

**需要明白，梯度矩阵和原矩阵的形状不变**

我们使用损失函数：
$$
L(w, b) = \frac{1}{2}(\hat y^{(i)}-y^{(i)})^2
$$

![image-20240425230323660](./Andrew_Ng _DL.assets/image-20240425230323660.png)

我们对如上神经网络进行反向传播

先看一下计算图

![image-20240422010922210](./Andrew_Ng _DL.assets/image-20240422010922210.png)

符合略微不同，但是问题不大

**以下式子中，我们省略样本编号$(i)$​**

例如：$A^{[3]} =  \begin{bmatrix}A^{[3]}_1 & A^{[3]}_2 \end{bmatrix}$，表示了单个样本的输出



- 输出层

  - $$
    dA^{[3]} = \frac{\partial L}{\partial A^{[3]}} = \begin{bmatrix}
    A^{[3]}_1-y_1 & A^{[3]}_2-y_2 
    \end{bmatrix}
    $$

  - $$
    dZ^{[3]} &= \frac{\partial L}{\partial Z^{[3]}} = \frac{\partial L}{\partial A^{[3]}}\frac{\partial A^{[3]}}{\partial Z^{[3]}}\\
    &=\begin{bmatrix}
    (A^{[3]}_1-y_1)\sigma'(Z^{[3]}_1) & (A^{[3]}_2-y_2)\sigma'(Z^{[3]}_2) )
    \end{bmatrix}
    $$

  - $$
    db^{[3]} = \frac{\partial L}{\partial Z^{[3]}} \frac{\partial Z^{[3]}}{\partial b^{[3]}} = \frac{\partial L}{\partial Z^{[3]}} \begin{bmatrix}
    1 \\ 1
    \end{bmatrix} = \begin{bmatrix}
    (A^{[3]}_1-y_1)\sigma'(Z^{[3]}_1)\\ (A^{[3]}_2-y_2)\sigma'(Z^{[3]}_2) )
    \end{bmatrix}  = dZ^{[3]T}
    $$

  - $$
    dw^{[3]} &=\frac{\partial L}{\partial Z^{[3]}} \frac{\partial Z^{[3]}}{\partial w^{[3]}} \\
    &= \begin{bmatrix}
    (A^{[3]}_1-y_1)\sigma'(Z^{[3]}_1)A^{[2]}_1 & (A^{[3]}_1-y_1)\sigma'(Z^{[3]}_1)A^{[2]}_2 & (A^{[3]}_1-y_1)\sigma'(Z^{[3]}_1)A^{[2]}_3 \\
    (A^{[3]}_2-y_2)\sigma'(Z^{[3]}_2)A^{[2]}_1 & (A^{[3]}_2-y_2)\sigma'(Z^{[3]}_2)A^{[2]}_2 & (A^{[3]}_2-y_2)\sigma'(Z^{[3]}_2)A^{[2]}_3
    \end{bmatrix} \\
    &=  \begin{bmatrix}
    (A^{[3]}_1-y_1)\sigma'(Z^{[3]}_1)\\ (A^{[3]}_2-y_2)\sigma'(Z^{[3]}_2) )
    \end{bmatrix} \begin{bmatrix}A_1^{[2]}  & A_2^{[2]}  & A_3^{[2]} \end{bmatrix} \\
    &= db^{[3]}A^{[2]}
    $$

- 隐藏层2

  - $$
    dA^{[2]} &= \frac{\partial L}{\partial Z^{[3]}}\frac{\partial Z^{[3]}}{\partial A^{[2]}} \\
    &=  \begin{bmatrix}dZ^{[3]}_1w^{[3]}_{1,1}+dZ^{[3]}_2w^{[3]}_{2,1} & dZ^{[3]}_1w^{[3]}_{1,2}+dZ^{[3]}_2w^{[3]}_{2,2} & dZ^{[3]}_1w^{[3]}_{1,3}+dZ^{[3]}_2w^{[3]}_{2,3} \end{bmatrix} \\
    &= \begin{bmatrix}
    (A^{[3]}_1-y_1)\sigma'(Z^{[3]}_1) & (A^{[3]}_2-y_2)\sigma'(Z^{[3]}_2) )
    \end{bmatrix} \begin{bmatrix}
    w^{[3]}_{1,1} & w^{[3]}_{1,2} & w^{[3]}_{1,3} \\
    w^{[3]}_{2,1} & w^{[3]}_{2,2} & w^{[3]}_{2,3} 
    \end{bmatrix} \\
    &= dZ^{[3]}w^{[3]}
    $$

  - $$
    dZ^{[2]} = \begin{bmatrix} dA^{[2]}_1 \sigma'(Z^{[2]}_1) & dA^{[2]}_2\sigma'(Z^{[2]}_2) & dA^{[2]}_3\sigma'(Z^{[2]}_3) \end{bmatrix}
    $$

  - $$
    db^{[2]} = \begin{bmatrix} dA^{[2]}_1 \sigma'(Z^{[2]}_1) \\dA^{[2]}_2\sigma'(Z^{[2]}_2) \\ dA^{[2]}_3\sigma'(Z^{[2]}_3) \end{bmatrix}= dZ^{[2]T}
    $$

  - $$
    dw^{[2]} = db^{[2]}A^{[1]T}
    $$
  
- 隐藏层1（我们假装前面还有一个输入层$A_0$没画出来，多推几层看看规律）

  - $$
    dA^{[1]} = dZ^{[2]}w^{[2]}
    $$

  - $$
    dZ^{[1]} = \begin{bmatrix} dA^{[1]}_1 \sigma'(Z^{[1]}_1) & dA^{[1]}_2\sigma'(Z^{[1]}_2)  \end{bmatrix}
    $$

- $$
  db^{[1]} = dZ^{[1]T}
  $$

- $$
  dw^{[1]} = db^{[1]}A^{[0]}
  $$

- 



整理一下：

- 求出$dA^{[L]}$

  - 输出层需要通过损失函数求导得到
  - 否则：$dA^{[L]} = dZ^{[L+1]}w^{[L+1]}$

- 求出$dZ^{[L]}$

  - 我们引入阿达玛乘积：$C = A\circ B $，有：$C_{i,j} = A_{i,j}B_{i,j}$
  - $dZ^{[L]} = dA^{[L]} \circ \sigma'(Z^{[L]})$​
- 求出$db^{[L]} = dZ^{[L]T}$
- 求出$dw^{[L]} = db^{[L]}A^{[L-1]}$​
  - $L>0$，输入层的$A^{[L = 0]}$即为输入样本




#### 反向传播：多个样本的计算

和上面已经没什么差别了

只有$A$和$dA$初始化时，从单个向量变成多维向量（矩阵）

例如：
$$
dA^{[3]} = \frac{\partial L}{\partial A^{[3]}} = \begin{bmatrix}
A^{[3](1)}_1-y_1^{(1)} & A^{[3](1)}_2-y_2 ^{(1)}\\
A^{[3](2)}_1-y_1^{(2)} & A^{[3](2)}_2-y_2 ^{(2)}\\
... & ...\\
A^{[3](n)}_1-y_1^{(n)} & A^{[3](n)}_2-y_2 ^{(n)}
\end{bmatrix}
$$
其他按原表达式继续进行即可

整理一下：

- 求出$dA^{[L]}$

  - 输出层需要通过损失函数求导得到
  - 否则：$dA^{[L]} = dZ^{[L+1]}w^{[L+1]}$
- 求出$dZ^{[L]}$

  - 我们引入阿达玛乘积：$C = A\circ B $，有：$C_{i,j} = A_{i,j}B_{i,j}$
  - $dZ^{[L]} = dA^{[L]} \circ \sigma'(Z^{[L]})$​
- 求出$db^{[L]} = dZ^{[L]T}\begin{bmatrix}1\\1\\.\\1\end{bmatrix}_{}$，
- 求出$dw^{[L]} = dZ^{[L]T}A^{[L-1]}$
  - $L>0$，输入层的$A^{[L = 0]}$即为输入样本

