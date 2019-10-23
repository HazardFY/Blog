# segmentation相关基础

主要参考博客：

https://blog.csdn.net/Biyoner/article/details/82591370

https://blog.csdn.net/say_hi_andhelloworld/article/details/94839562

 https://blog.csdn.net/dugudaibo/article/details/83109814 

## 语意分割的定义

 语义分割是计算机视觉中十分重要的领域，它是指像素级地识别图像，即标注出图像中每个像素所属的对象类别。下图为语义分割的一个实例，其目标是预测出图像中每一个像素的类标签。 

 ![img](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-17-at-7.42.16-PM.png) 

语义分割问题的发展：

语义分割任务最初流行的深度学习方法是图像块分类（patch classification），即利用像素周围的图像块对每一个像素进行独立的分类。使用图像块分类的主要原因是分类网络通常是**全连接层**（full  connected  layer），且要求固定尺寸的图像。

2014 年，加州大学伯克利分校的 Long等人提出**全卷积网络（FCN）**，这使得卷积神经网络无需全连接层即可进行密集的像素预测，CNN 从而得到普及。使用这种方法可生成任意大小的图像分割图，且该方法比图像块分类法要快上许多。之后，语义分割领域几乎所有先进方法都采用了该模型。

除了全连接层，使用卷积神经网络进行语义分割存在的另一个大问题是**池化层**。池化层虽然扩大了感受野、聚合语境，但因此造成了**位置信息的丢失**。但是，语义分割要求类别图完全贴合，因此需要保留位置信息。 有两种不同结构来解决该问题。 第一个是**编码器--解码器**结构。编码器逐渐减少池化层的空间维度，解码器逐步修复物体的细节和空间维度。编码器和解码器之间通常存在快捷连接，因此能帮助解码器更好地修复目标的细节。U-Net是这种方法中最常用的结构。   第二种方法使用**空洞/带孔卷积**（dilated/atrous  convolutions）结构，来去除池化层。

## encoder-decoder

 针对语义分割任务构建神经网络架构的最简单的方法是简单地堆叠多个卷积层（使用same填充以保留需要的维度）并输出最终的分割图。这通过特征映射的连续变换直接学习从输入图像到其对应分割的映射。但在整个网络中保持全分辨率的计算成本非常高 \

 ![img](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-19-at-12.32.20-PM.png) 

 对于深度卷积网络，**浅层主要学习低级的信息**，随着网络越深，学习到更高级的特征映射。**为了保持表达能力，我们通常需要增加特征图的数量（通道数），从而可以得到更深的网络**。对于图像分类来说，由于我们只关注图像“是什么”（而不是位置在哪），因而我们可以通过阶段性对特征图降采样（downsampling）或者带步长的卷积（例如，压缩空间分辨率）。然而对于图像分割，我们希望我们的模型产生全分辨率语义预测。 

 **编码器/解码器结构**，其中我们对输入的空间分辨率进行下采样，生成分辨率较低的特征映射，它能高效地进行分类。随后，上采样可以将特征还原为全分辨率分割图 

 ![img](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-16-at-10.33.29-PM.png) 

### 上采样的方法：（也有叫反池化的）

![img](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-19-at-12.54.50-PM.png) 

可学习的上采样

 ![img](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-19-at-3.12.51-PM.png) 

对于转置卷积，我们从低分辨率特征图中获取单个值，并将滤波器中的所有权重乘以该值，将这些加权值投影到输出要素图中 

 ![img](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-21-at-11.01.29-PM.png) 

 它所能做的只是把feature map还原到本来的大小，并不是真的能逆转卷积的运算过程。

### **卷积和反卷积的详细讲解：**

 ![img](https://img-blog.csdn.net/20181017220744712?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1Z3VkYWlibw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

上图中最左边的是一个卷积核，中间的是一个原始的卷积过程，蓝色部分 (44) 是输入 input feature map ，而绿色部分 (22) 是输出 output feature map 部分，其中深蓝色对应一个卷积核的大小 (3*3) ，上面的卷积的过程共经历 4 次卷积。

  假如我们将原始图像按照上图中**最右边所标注的顺序展开为列向量**，并记为向量 X ，将得到的向量式的 feature map 作为输出并记为 Y，则卷积的过程可以表示如下的这种形式
																		Y=CX
 ![img](https://img-blog.csdn.net/20181017215726823?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1Z3VkYWlibw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

以矩阵 C 的第一行为例，其实非零权重的位置与上图左右边标注的位置是一样的；同理第二行与第二次卷积运算是一样的……。经过上面的运算我们会得到一个 4×1 大小的向量，将这个向量按照展开图像的反向顺序，重构成为一个矩阵，即可得到卷积所对应的输出。

  这样，卷积的过程就转变为了一个稀疏的权重矩阵与一个图像向量相乘的过程。
 ![img](https://img-blog.csdn.net/20181019104051219?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1Z3VkYWlibw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

所以我们可以知道，通过卷积核可以定义两个矩阵，一个是对应前向传播的矩阵 C ，另一个是对应反向传播的矩阵 C^T

 	**转置卷积是通过交换前向传播与反向传播得到的。**核心思想是交换C矩阵。 **总可以使用直接卷积来模拟转置卷积**，但是由于要在行列之间补零，所以执行的效率会低。通过后面的分析，我们也可以认为，**转置卷积实际上就是卷积**，将小的feature map先进行padding再进行卷积。 

​	计算：
$$
W_2=(W_1-F+2P)/S+1
$$
转置卷积的话计算如下：
$$
W_1=S(W_2-1)-2P+F
$$
下列是具体例子：

 ![img](https://img-blog.csdn.net/20181018104956447?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1Z3VkYWlibw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

​	如上图所示，第一行是卷积的过程，第二行是一个反卷积的过程。卷积和转置卷积是一对相关的概念，转置卷积嘛，你总得告诉我你针对谁是转置卷积啊。在上图中，卷积过程是在一个 4×4 大小，padding =0 的 feature map 上使用一个 3×3 大小的卷积核进行 s=1s=1s=1 的卷积计算，输出的 output feature map 大小为 2×2
​	这个时候W1计算出来是4，但是padding的值需要下以这种方式计算：
$$
P^T=F-P-1
$$
其中P^T是转置卷积中padding的大小，F是直接卷集中卷积核的大小，P是直接卷积中padding的大小

* 步长小于1的转置卷积

   由于转置卷积的步长是直接卷积的倒数，因此当直接卷积的步长 s>1 的时候，那么转置卷积的步长就会是分数 ，通过以下例子来了解

 ![img](https://img-blog.csdn.net/20181018153854944?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2R1Z3VkYWlibw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

如上图是一个输入 feature map 为 5×5，卷积核大小为 3×3，步长 s=2 的直接卷积的转置卷积，此时的转置卷积的输入是在 2×2的矩阵间进行插孔得到的。首先计算此时转置卷积输出的大小，我们发现与之前的计算方法是一样的
$$
W_1=S(W_2-1)-2P+F=2\times (2-1)-2\times0+3=5
$$
计算padding的大小
$$
P^T=F-P-1=3-0-1=2
$$
如何体现出步长是分数步长：在原始的卷积红插入数字0 ，这使得内核比以往单位步幅移动的速度慢，具体在每两个元素之间插入$s-1$个0此时的转置卷积输入尺寸大小由$W_2$变为$W_2+(W_2-2)(s-1)$.



**反激活**： 我们在Alexnet中，relu函数是用于保证每层输出的激活值都是正数，因此对于反向过程，我们同样需要保证每层的特征图为正值，也就是说这个反激活过程和激活过程没有什么差别，都是直接采用relu函数 





## VGG16

VGG16网络架构：

 ![img](https://img-blog.csdn.net/20180725174138144?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5ODkzMzg1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

 进行第一个卷积之后得到224×224×64的特征图，接着还有一层224×224×64，得到这样2个厚度为64的卷积层，意味着我们用64个过滤器进行了两次卷积。正如我在前面提到的，这里采用的都是大小为3×3，步幅为1的过滤器，并且都是采用**same**卷积， 之后通过pooling 层对图像进行放缩。

 **VGG-16**的这个数字16，就是指在这个网络中包含16个卷积层和全连接层 

**优点**：结构简单，基本上都是图像size的缩小以及通道数的翻倍，具有一致性

**缺点**：需要训练的特征数非常大



## FCN

主要贡献：

- 为语义分割引入了 [端到端](https://www.zhihu.com/question/51435499/answer/129379006) 的**全**卷积网络，并流行开来

- 重新利用 ImageNet 的预训练网络用于语义分割

- 使用 反卷积层 进行上采样

- 引入跳跃连接来改善上采样粗糙的像素定位

  

将经典的CNN网络中最后的全连接+softmax输出替换成卷积层，输出是一张已经label好的图片。

与经典的CNN在卷积层之后使用全连接层得到固定长度的特征向量进行分类（全联接层＋softmax输出）不同，FCN可以接受任意尺寸的输入图像，采用反卷积层对最后一个卷积层的feature map进行上采样, 使它恢复到输入图像相同的尺寸，从而可以对每个像素都产生了一个预测, 同时保留了原始输入图像中的空间信息, 最后在上采样的特征图上进行逐像素分类。

最后逐个像素计算softmax分类的损失, 相当于每一个像素对应一个训练样本。下图是Longjon用于语义分割所采用的全卷积网络(FCN)的结构示意图：

 ![FCN结构](http://img.blog.csdn.net/20161022111939034) 

## SegNet

**主要贡献**：

*  最大池化指数被转移到解码器中，改善了分割的效果 

​         在结构上看，SegNet和U-net其实大同小异，都是编码-解码结果。区别在意，SegNet没有直接融合不同尺度的层的信息，为了解决为止信息丢失的问题，SegNet使用了带有坐标（index）的池化。如下图所示，在Max pooling时，选择最大像素的同时，记录下该像素在Feature map的位置（左图）。在反池化的时候，根据记录的坐标，把最大值复原到原来对应的位置，其他的位置补零。后面的卷积可以把0的元素给填上。这样一来，就解决了由于多次池化造成的位置信息的丢失。 

 ![img](http://simonduan.site/img/passage/20170723/006.jpg) 

## U-Net

**主要贡献**

- 改进了FCN，把扩展路径完善了很多，多通道卷积与类似FPN（特征金字塔网络）的结构相结合。
- 利用少量数据集进行训练测试，为医学图像分割做出很大贡献。
- 通过对每个像素点进行分类，获得更高的分割准确率。
- 用训练好的模型分割图像，速度快

U-net与其他常见的分割网络有一点非常不同的地方：U-net采用了完全不同的特征融合方式：拼接，U-net采用将特征在channel维度拼接在一起，形成更厚的特征。而FCN融合时使用的对应点相加，并不形成更厚的特征。

所以语义分割网络在特征融合时有两种办法：

-  FCN式的对应点相加，对应于TensorFlow中的tf.add()函数；
- U-net式的channel维度拼接融合，对应于TensorFlow的tf.concat()函数，比较占显存。

5个pooling layer实现了网络对图像特征的多尺度特征识别。
上采样部分会融合特征提取部分的输出，这样做实际上是将多尺度特征融合在了一起，以最后一个上采样为例，它的特征既来自第一个卷积block的输出(同尺度特征)，也来自上采样的输出(大尺度特征)，这样的连接是贯穿整个网络的，你可以看到上图的网络中有四次融合过程，相对应的FCN网络只在最后一层进行融合

 ![img](http://simonduan.site/img/passage/20170723/003.jpg) 

##  DeepLab

DeepLab有v1 v2 v3，第一篇名字叫做DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs。这一系列论文引入了以下几点比较重要的方法：

第一个是带洞卷积，英文名叫做Dilated Convolution，或者Atrous Convolution。带洞卷积实际上就是普通的卷积核中间插入了几个洞，如下图。

 ![img](https://img-blog.csdn.net/20180529132835375?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0p1bGlhbG92ZTEwMjEyMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

 它的运算量跟普通卷积保持一样，好处是它的“视野更大了”，比如普通3x3卷积的结果的视野是3x3，插入一个洞之后的视野是5x5。视野变大的作用是，在特征图缩小到同样倍数的情况下可以掌握更多图像的全局信息，这在语义分割中很重要 

## Pyramid Scene Parsing Network

 Pyramid Scene Parsing Network的核心贡献是Global Pyramid Pooling，翻译成中文叫做全局金字塔池化。它将特征图缩放到几个不同的尺寸，使得特征具有更好地全局和多尺度信息，这一点在准确率提升上上非常有用 

 ![img](https://img-blog.csdn.net/20180626214856849?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0p1bGlhbG92ZTEwMjEyMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

## Mask R-CNN

Mask R-CNN是大神何凯明的力作，将Object Detection与Semantic Segmentation合在了一起做。它的贡献主要是以下几点。

第一，神经网络有了多个分支输出。Mask R-CNN使用类似Faster R-CNN的框架，Faster R-CNN的输出是物体的bounding box和类别，而Mask R-CNN则多了一个分支，用来预测物体的语义分割图。也就是说神经网络同时学习两项任务，可以互相促进。

第二，在语义分割中使用Binary Mask。原来的语义分割预测类别需要使用0 1 2 3 4等数字代表各个类别。在Mask R-CNN中，检测分支会预测类别。这时候分割只需要用0 1预测这个物体的形状面具就行了。

第三，Mask R-CNN提出了RoiAlign用来替换Faster R-CNN中的RoiPooling。RoiPooling的思想是将输入图像中任意一块区域对应到神经网络特征图中的对应区域。RoiPooling使用了化整的近似来寻找对应区域，导致对应关系与实际情况有偏移。这个偏移在分类任务中可以容忍，但对于精细度更高的分割则影响较大。

为了解决这个问题，RoiAlign不再使用化整操作，而是使用线性插值来寻找更精准的对应区域。效果就是可以得到更好地对应。实验也证明了效果不错。下面展示了与之前方法的对比，下面的图是Mask R-CNN，可以看出精细了很多。