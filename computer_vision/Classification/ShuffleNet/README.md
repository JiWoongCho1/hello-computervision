## ShuffleNet

The most accuracy CNNs ususally have hundreds of layers and thousands of channels, thus requiring computation at billions of FLOPs. This paper examines the opposite exetreme: pursuing the best accuracy in very limited computational budgets at tens or hundreds of MFLOPs, focusing on common mobile platforms such as drones, robots, and smartphones. Authors propse using _pointwise gropu convolutions_ to reduce computation complexity of 1x1 convolutions. To overcome the side effects brought by group convolutions, they come up with a novel _channel shuffle_ operation to help the information flowing across feature channels. 

#### Group Convolution

The concept of group convolution, which was first introduced in _AlexNet_ for distributing the model over two GPUs, has been well domonstated its effectiveness in ResNeXt. They work generalizes group convolution and depthwise seperable convolution in a novel form

#### Channel shuffle operation

To the best of their knowledge, the idea of channel shuffle operation is rarely mentioned in previous work on efficient model design, although CNN library _cuda-convnet_  supports 'random sparse convolution' layer, which is equivalent to random channel shuffle followed by a group convolutional layer. By ensuring that each convolutional operates only on the corresponding input channel group, group convolution significantly reduces computation cost. However, if multiple group convolutions stack together, there is one side effect is that outputs ffrom a certain channel are only derived from a small fraction of input channels. This property blocks information flow between channel groups and weakens representation. 

So they can first divide the channels in each group into several subgroups. This can be efficiently and elegantly implemented by a _channel shuffle_ operation. Taking advantage of the channel shuffle operation, they propose a novel _ShuffleNet_ unit specially designed for small networks. The purpose of shuffle operation is to enable cross-group information flow for multiple group convolution layers. They compared the performance of group numbers with channel shuffle.

In addition, in ShuffleNet depthwise convolution only performs on bottleneck feature maps. Even though depthwise convolution usually has very low theoretical complexity, they find it difficult to efficiently implemen on low power devices, which may result from a worse computation/memory access ratio compared with other dense operations.

From the results, they see that models with group convolutios consistently perform better than the counterparts without pointwise grou convolutions.
