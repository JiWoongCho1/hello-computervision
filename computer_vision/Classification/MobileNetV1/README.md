## MobileNetV1


The general trend has been to make deeper and more complicated networks in order to achieve higher accuracy. However these advantages to imporve accuracy are not necessarily making networks more efficient with respect to size and speed. So in this paper, they propse a new architecture(but it is similar to XceptionNet) that is more efficient, small and low latency called MobileNet.

Model is based on depthwise seperable convolutions which consists of depthwise convolution and pointwise convolution. This factorization has the effect of drastically reducing computation and model size. The standard convolution operation has the effect of filtering features based on the convolutional kernels and combining features in order to produce a new representation. Depthwise convolution is extremely efficient relative to standard convolution but it only filters input channels, it does not combine them to create new features. So an additional layer that computes a linear combination of the output of depthwise convolution via 1x1 convolution is needed in order to generate these new features.  

The MobileNet structure is built on depthwise separable convolutions except for the first layer which is a fully convolution. Additionally they introduced a very simple parameters a,b (alpha, beta). Alpha is the multiplier of input channels so reduce the channels and beta is the multiplier of the image resolution. Authors find and experiment the models that change the parameters(alpha and beta).

Concludely MobileNet is nearly as accurate as VGG16 while being 32 times smaller and 27 times less compute intensive. It is more accurate than GoogleNet while being smaller and more than 2.5 times less computation. And MobileNet-based on classifier is resilient to aggressive model shrinking. It achieves a similar mean average precision across attributes as the in-hours while consuming only 1% the Multi-Adds. And it is also effective n modern object detection systems.
