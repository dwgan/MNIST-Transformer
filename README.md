# MNIST-PyTorch

学习深度学习已经有好几个月了，从零开始搭建开发环境，维护服务器，了解各个研究方向，常用的模型都见过了，在深度压缩感知方向也有了一定深度的了解，目前也能抽象出一些问题。然而，在实际coding的时候还是感觉力不从心，归根结底是基础不牢。所以，今天打算针对MNIST任务手撕一个CNN网络，进一步巩固基础。

## MNIST任务

The MNIST database (**Modified National Institute of Standards and Technology database**) is a large database of handwritten digits that is commonly used for training various image processing systems.

这是一个根据图像识别手写数字的任务，输入是28*28像素的图像，输出则是0-9的数字。

![MNIST sample images](https://upload.wikimedia.org/wikipedia/commons/f/f7/MnistExamplesModified.png)

MNIST 包括6万张28x28的训练样本，1万张测试样本。

## 解决思路

针对MNIST任务有以下几种经典网络：

1. **LeNet-5**: 这是最著名的早期卷积神经网络之一，由Yann LeCun于1998年开发。它专为手写数字识别设计，因此非常适合于MNIST任务。LeNet-5使用了卷积层、池化层（下采样层）和全连接层的结构。
2. **AlexNet**: 虽然它是为ImageNet挑战赛设计的，但AlexNet的结构也可以应用于MNIST，尤其是当你需要一个更深层次的网络来实验或提高识别精度时。AlexNet使用了ReLU激活函数、多个卷积层、池化层和全连接层。
3. **VGGNet**: 由牛津大学的Visual Geometry Group开发，特别是VGG-16和VGG-19因其架构简洁而广受欢迎。这些网络通过重复使用相同的卷积层结构来增加网络深度。
4. **ResNet (Residual Networks)**: 由微软研究院提出，赢得了2015年ImageNet竞赛。ResNet通过引入残差连接来解决深度神经网络中的梯度消失问题，使得网络可以成功地深化到数百甚至数千层。

为了加深对于网络的理解，这里将自定义一个基于CNN的网络。其网络架构如下：

1. 首先明确任务的输入输出。输入是n个$28\times28$像素的图像（维度是$28\times28=784$），输出是数字0-9（维度是10）。
2. 确定网络类型。因为输出是图像信号，因此可以采用卷积层和池化层对图像进行特征提取；输出是特征类别，因此可以采用全连接层进行特征分类。
3. 激活函数。引入激活函数可以使得神经网络具有非线性，从而具备更强的表示能力，这里将采用最简单的ReLU函数。
4. 优化器。PyTorch提供了一个训练网络的框架，用户只需要调用其接口即可对网络进行前向传播和反向传播。

训练过程：

第2、3步完成了网络的定义后，需要给神经网络输入训练数据，通过“学习”得到适配该任务的实际网络参数。一个训练过程如下：用户提供标准的输入数据（手写图像）和对应输出数据（标签0-9），数据输入网络后通过前向传播得到一个输出，计算该输出和标准输出之间的误差，调用框架利用链式法则计算网络的梯度，在反向传播过程根据学习率调整网络参数。再输出下一组数据进行学习，以此类推。

深度学习方法和传统方法最大的区别就在于，传统方法是模型驱动的，而深度学习是数据驱动的。即在模型架构设置好之后，网络参数是根据输出的数据自动更新的，用户无需干预。用户需要做的事就是选择合适的数据以及调整模型迭代的速度，即调整学习率（太大了容易振荡，太小了更新很慢浪费时间），俗称调参（调参调的是超参数）。

## 网络设计和网络分析

#### 首先建立工程

工程结构如图所示，其中

<img src="https://raw.githubusercontent.com/dwgan/PicGo/main/img/202401152315517.png" alt="image-20240115231523449" style="zoom:50%;" />

data_loader负责加载数据，代码如下：

```python
import torch
from torchvision import datasets,transforms

def train_loader(batch_size):
    return torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

def test_loader(batch_size):
    return torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
```

model定义了我的模型结构，代码如下：

```python
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5) # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = nn.Conv2d(10, 20, 3) # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20*10*10, 500) # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(500, 10) # 输入通道数是500，输出通道数是10，即10分类
    def forward(self,x):
        in_size = x.size(0) # batch_size=512，x是512*1*28*28的tensor
        out = self.conv1(x) # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = F.relu(out) # batch*10*24*24（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2) # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = self.conv2(out) # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out) # batch*20*10*10
        out = out.view(in_size, -1) # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = self.fc1(out) # batch*2000 -> batch*500
        out = F.relu(out) # batch*500
        out = self.fc2(out) # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1) # 计算log(softmax(x))
        return out
```

train包含了训练函数，代码如下：

```python
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) #加载数据，data是图像，target是其对应的数字
        output = model(data) #前向传播，将每一张28*28像素的图像映射成10个数字，最终第几个数字的值最大这个图片就是手写的几
        loss = F.nll_loss(output, target) #计算当前损失，即衡量按照当前的模型对图像进行分类得到的结果和目标差多少
        optimizer.zero_grad() #清除累积梯度，PyTorch中默认是会累积梯度的，累积梯度对于RNN网络计算很友好
        loss.backward() #进行反向传播，通过链式法则求梯度
        optimizer.step() #进行梯度更新，根据学习率设置的迭代步长进行梯度下降
        if(batch_idx+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
```

test包含了测试函数，代码如下：

```python
import torch
import torch.nn.functional as F

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

main是主循环，代码如下：

```python
import torch
import torch.optim as optim
from model import ConvNet
from train import train
from test import test
from data_loader import train_loader, test_loader

batch_size = 512 #每一批数据为512个图像
epochs = 20 #对所有样本训练20轮
learning_rate = 1e-4 #学习率对应了反向传播中梯度下降的步长
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #如果可用则采用GPU进行加速
model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    # 对模型进行一轮训练
    train(model, device, train_loader(batch_size), optimizer, epoch)
    # 测试该轮训练之后的模型性能
    test(model, device, test_loader(batch_size))
```

其中最主要的循环是训练过程，即train函数，其主要包括数据加载、前向传播、计算损失、反向传播更新模型等几个步骤。

最重要最核心的部分则是模型的定义和前向传播过程，即model函数，它决定了整个神经网络的结构。为了更加清晰地理解信号在神经网络中是如何传播地，这里对一个前向传播过程进行深入分析。

断点来到前向传播函数

![image-20240115233248483](https://raw.githubusercontent.com/dwgan/PicGo/main/img/202401152332562.png)

在命令行中输入如下代码可以将变量进行可视化

```python
import matplotlib.pyplot as plt
for i in range(20):
    for j in range(1):
        plt.subplot(4, 5, i*1+j+1)
        img = x[i, j, :, :]
        img = img.cpu().detach().numpy()
        plt.imshow(img)
        plt.axis('off')
plt.show()
```

可以看到，这里地x是$512\times1\times28\times28$的tensor，取其前20个输出观察，它们是20个手写数字地图像

![image-20240115224848103](https://raw.githubusercontent.com/dwgan/PicGo/main/img/202401152248185.png)

代码执行到下一行，继续将out可视化，可以看到卷积层1输入通道是1输出是10，它的物理含义是将一幅图像进行了10次特征提取，由于这十个CNN特征模板是随机生成的，因此提取方式也很随机

<img src="https://raw.githubusercontent.com/dwgan/PicGo/main/img/202401152257725.png" alt="image-20240115225715658" style="zoom:200%;" />

继续下一行，经过激活函数，可以看到图像特征变得明显了

<img src="https://raw.githubusercontent.com/dwgan/PicGo/main/img/202401152258436.png" alt="image-20240115225829396" style="zoom:200%;" />

下一行，对图像进行池化，更加抽象了

<img src="https://raw.githubusercontent.com/dwgan/PicGo/main/img/202401152259690.png" alt="image-20240115225900647" style="zoom:200%;" />

下一行，经过第二次卷积之后人眼已经很难分辨出来单个图像的特征了，不过通道数变多了，对于计算机而言理论上是有更多特征的

<img src="https://raw.githubusercontent.com/dwgan/PicGo/main/img/202401152300279.png" alt="image-20240115230019086" style="zoom:200%;" />

往下一行，激活之后会使得信号更加稀疏，不过人眼已经很难提取特征了

<img src="https://raw.githubusercontent.com/dwgan/PicGo/main/img/202401152300256.png" alt="image-20240115230053225" style="zoom:200%;" />

后面是将图像经过全连接层展开，最后变成10个类别，10个数字的大小表示和对应数字的相似度

![image-20240115230327578](https://raw.githubusercontent.com/dwgan/PicGo/main/img/202401152303622.png)



## 总结

利用神经网络进行图像分类的原理，其实就是通过合理构建一个复杂的映射关系（神经网络），然后通过有效的数据去找到映射关系的准确形式（学习和调参），当所有可能的类型都被其“学习”之后，它就具有了自主判断的能力。

回顾一下人类小孩学习的过程，只有见过足够多（训练数据量），才能正确区分对和错（实现分类任务），每一次犯错的时候都要经受父母的混合双打（计算损失函数），然后下一次才能做得更好（梯度下降更新迭代），区分不同学习能力的其实是学习的方法（模型架构），以及知错能改的态度（误差反馈机制），不禁恍然大悟



### 工程源码

https://github.com/dwgan/MNIST-PyTorch

### 参考文献

https://blog.csdn.net/sikh_0529/article/details/126901302
