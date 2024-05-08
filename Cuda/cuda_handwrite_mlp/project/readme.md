# Pytorch验证及数据集配置
先运行一遍/pytorch-check/sgd.ipynb
会把数据集进行下载，并且将预处理好的数据集存储到/dataset/下

# C++

### 参数设定

```cpp
	const int num_examples = 60000;     // 训练集大小
	const int test_num_examples = 10000; // 测试集大小
    const int len_X = 28*28, len_Y = 10; // 单个输入数据的维度与输出数据的维度

    const double lr = 0.01;         // 学习率 报nan的时候调低这里
    const int batch_size = 1024;    // 
    const int num_epochs = 100;     // 训练次数
    const int check_epochs = 10;    // 每10次输出一次loss


    std::ifstream input("../dataset/mnist");        // 训练集   x_0 x_1 x_2 ... x_len y_0 y_1 ... y_len
    std::ifstream input2("../dataset/mnist_test");      // 测试集


    MLP mlp(lr, batch_size, test_num_examples, "CrossEntropyLoss"); 
        // 损失函数： "CrossEntropyLoss" 请于softmax搭配使用 
        // 梯度计算使用的是 交叉熵和softmax的快捷计算 

    mlp.setInputLayer(len_X); // 输入层
    mlp.addLayer(128, "ReLU");  // 激活函数
    mlp.addLayer(64, "ReLU");
    mlp.addLayer(10, "softmax"); // 输出层

```

### 编译运行

```bash
cd /build
cmake ..
make -j3
./app
```

