

# Resnet 网络结构

## 1.这一句的格式

```Python
nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)
```

似乎是直接对应in_planes，out_planes输入，输出通道的个数

**也是一种格式，一般用于网络中间，图片输入的规格自动计算，只需要输入输出通道数，但是初始输入的图片格式还是要的**

## 2. Resnet基础网络的搭建

![img](https://img-blog.csdn.net/20180319160714266?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L2ppYW5ncGVuZzU5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

从中可以看出Resnet网络是高度模块化的，其中最主要的模块如下：

![img](https://www.aiuai.cn/uploads/sina/5ce8dfe46e373.jpg)

```

###
与基础版的不同之处只在于这里是三个卷积，分别是1x1,3x3,1x1,分别用来压缩维度，卷积处理，恢复维度，
inplane是输入的通道数，plane是输出的通道数，expansion是对输出通道数的倍乘，在basic中expansion是1，
此时完全忽略expansion这个东东，输出的通道数就是plane，然而bottleneck就是不走寻常路，它的任务就是要对
通道数进行压缩，再放大，于是，plane不再代表输出的通道数，而是block内部压缩后的通道数，输出通道数变为
plane*expansion。接着就是网络主体了。
###
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

```

```
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
```

ResNet网络构建：

```
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(512 * block.expansion, 128, kernel_size=1, stride=1, 
                               bias=False)
        self.fc1 = nn.Linear(128 * 48, 128)
        self.fc2 = nn.Linear(128, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
	#根据blocks(block的数量)构建多个block块
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
        #这一步的目的，downsample用于前面的计算初始输入的残差residual,并和卷积层输出相加
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
		#一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
		，同时以神经网络模块为元素的有序字典也可以作为传入参数
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.maxpool2(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.fc2(x)

        return x
```

ResNet 18的网络结构：这是resnet18,只贴出了前两层，其他层类似，第一层是没有downsample的，因为输入与输出通道数一样，其余层都有downsample

```
ResNet(
  ``(conv1): Conv2d(``3``, ``64``, kernel_size``=``(``7``, ``7``), stride``=``(``2``, ``2``), padding``=``(``3``, ``3``), bias``=``False``)
  ``(bn1): BatchNorm2d(``64``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
  ``(relu): ReLU(inplace)
  ``(maxpool): MaxPool2d(kernel_size``=``3``, stride``=``2``, padding``=``1``, dilation``=``1``, ceil_mode``=``False``)
  ``(layer1): Sequential(
    ``(``0``): BasicBlock(
      ``(conv1): Conv2d(``64``, ``64``, kernel_size``=``(``3``, ``3``), stride``=``(``1``, ``1``), padding``=``(``1``, ``1``), bias``=``False``)
      ``(bn1): BatchNorm2d(``64``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(relu): ReLU(inplace)
      ``(conv2): Conv2d(``64``, ``64``, kernel_size``=``(``3``, ``3``), stride``=``(``1``, ``1``), padding``=``(``1``, ``1``), bias``=``False``)
      ``(bn2): BatchNorm2d(``64``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
    ``)
    ``(``1``): BasicBlock(
      ``(conv1): Conv2d(``64``, ``64``, kernel_size``=``(``3``, ``3``), stride``=``(``1``, ``1``), padding``=``(``1``, ``1``), bias``=``False``)
      ``(bn1): BatchNorm2d(``64``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(relu): ReLU(inplace)
      ``(conv2): Conv2d(``64``, ``64``, kernel_size``=``(``3``, ``3``), stride``=``(``1``, ``1``), padding``=``(``1``, ``1``), bias``=``False``)
      ``(bn2): BatchNorm2d(``64``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
    ``)
  ``)
  ``(layer2): Sequential(
    ``(``0``): BasicBlock(
      ``(conv1): Conv2d(``64``, ``128``, kernel_size``=``(``3``, ``3``), stride``=``(``2``, ``2``), padding``=``(``1``, ``1``), bias``=``False``)
      ``(bn1): BatchNorm2d(``128``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(relu): ReLU(inplace)
      ``(conv2): Conv2d(``128``, ``128``, kernel_size``=``(``3``, ``3``), stride``=``(``1``, ``1``), padding``=``(``1``, ``1``), bias``=``False``)
      ``(bn2): BatchNorm2d(``128``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(downsample): Sequential(
        ``(``0``): Conv2d(``64``, ``128``, kernel_size``=``(``1``, ``1``), stride``=``(``2``, ``2``), bias``=``False``)
        ``(``1``): BatchNorm2d(``128``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``)
    ``)
    ``(``1``): BasicBlock(
      ``(conv1): Conv2d(``128``, ``128``, kernel_size``=``(``3``, ``3``), stride``=``(``1``, ``1``), padding``=``(``1``, ``1``), bias``=``False``)
      ``(bn1): BatchNorm2d(``128``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(relu): ReLU(inplace)
      ``(conv2): Conv2d(``128``, ``128``, kernel_size``=``(``3``, ``3``), stride``=``(``1``, ``1``), padding``=``(``1``, ``1``), bias``=``False``)
      ``(bn2): BatchNorm2d(``128``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
    ``)
  ``)
```

这是resnet50，只贴出了第一层，每一层都有downsample，因为输出与输入通道数都不一样。可以看在resnet类中输入的64，128，256，512，都不是最终的输出通道数，只是block内部压缩的通道数，实际输出通道数要乘以expansion，此处为4。

```
ResNet(
  ``(conv1): Conv2d(``3``, ``64``, kernel_size``=``(``7``, ``7``), stride``=``(``2``, ``2``), padding``=``(``3``, ``3``), bias``=``False``)
  ``(bn1): BatchNorm2d(``64``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
  ``(relu): ReLU(inplace)
  ``(maxpool): MaxPool2d(kernel_size``=``3``, stride``=``2``, padding``=``1``, dilation``=``1``, ceil_mode``=``False``)
  ``(layer1): Sequential(
    ``(``0``): Bottleneck(
      ``(conv1): Conv2d(``64``, ``64``, kernel_size``=``(``1``, ``1``), stride``=``(``1``, ``1``), bias``=``False``)
      ``(bn1): BatchNorm2d(``64``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(conv2): Conv2d(``64``, ``64``, kernel_size``=``(``3``, ``3``), stride``=``(``1``, ``1``), padding``=``(``1``, ``1``), bias``=``False``)
      ``(bn2): BatchNorm2d(``64``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(conv3): Conv2d(``64``, ``256``, kernel_size``=``(``1``, ``1``), stride``=``(``1``, ``1``), bias``=``False``)
      ``(bn3): BatchNorm2d(``256``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(relu): ReLU(inplace)
      ``(downsample): Sequential(
        ``(``0``): Conv2d(``64``, ``256``, kernel_size``=``(``1``, ``1``), stride``=``(``1``, ``1``), bias``=``False``)
        ``(``1``): BatchNorm2d(``256``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``)
    ``)
    ``(``1``): Bottleneck(
      ``(conv1): Conv2d(``256``, ``64``, kernel_size``=``(``1``, ``1``), stride``=``(``1``, ``1``), bias``=``False``)
      ``(bn1): BatchNorm2d(``64``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(conv2): Conv2d(``64``, ``64``, kernel_size``=``(``3``, ``3``), stride``=``(``1``, ``1``), padding``=``(``1``, ``1``), bias``=``False``)
      ``(bn2): BatchNorm2d(``64``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(conv3): Conv2d(``64``, ``256``, kernel_size``=``(``1``, ``1``), stride``=``(``1``, ``1``), bias``=``False``)
      ``(bn3): BatchNorm2d(``256``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(relu): ReLU(inplace)
    ``)
    ``(``2``): Bottleneck(
      ``(conv1): Conv2d(``256``, ``64``, kernel_size``=``(``1``, ``1``), stride``=``(``1``, ``1``), bias``=``False``)
      ``(bn1): BatchNorm2d(``64``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(conv2): Conv2d(``64``, ``64``, kernel_size``=``(``3``, ``3``), stride``=``(``1``, ``1``), padding``=``(``1``, ``1``), bias``=``False``)
      ``(bn2): BatchNorm2d(``64``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(conv3): Conv2d(``64``, ``256``, kernel_size``=``(``1``, ``1``), stride``=``(``1``, ``1``), bias``=``False``)
      ``(bn3): BatchNorm2d(``256``, eps``=``1e``-``05``, momentum``=``0.1``, affine``=``True``, track_running_stats``=``True``)
      ``(relu): ReLU(inplace)
    ``)
  ``)
```