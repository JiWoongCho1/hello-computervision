## XceptionNet

This paper is based on the 'Inception' hypothesis that would independently look at cross-channel orrelations and at spatial correlations. The typical Inception module first looks at cross-channel correlations via a set of 1x1 convolutions, mapping the data into 3 or 4 separate spaces.And cross-channel correlations and spatial correlations are sufficiently decoupled that it is preferable not to map them jointly. Name of Xception is an 'extreme' version of an Inception module, based on the hypothesis. 

![extreme version of module](https://user-images.githubusercontent.com/90513931/214254721-80ee1e4b-94b4-4fa1-863a-95ca71b5d5d4.png)

The authors remarked that this extreme form of an Inception module is almost identical to a _depthwise separable convolution_. A spatial convolution performed independently over each channel of an input, followed by a _pointwise convolution_ that is a 1x1 convolution projecting the channels output by the depthwise convolution onto a new channel space.


The Xception achitecture has 36 convolutional layers forming the feature extraction base of the network and is a linear stack of depthwise separable convolution layers with residual connections. This makes the architecture very easy to define and mofify. For simplicity, they choose not to include auxilary classifier that is included in  Inception layers helping the backpropagation. This architecture shows a much larger performance improvement of the JPT dataset that is dataset of Google compared to the ImageNet dataset.

![architecture](https://user-images.githubusercontent.com/90513931/214254718-17deb7cb-75ab-4117-9d02-4793af30baec.png)

However there is no reason to believe that depthwise separable convolutions are optimal. it may be that intermediate points on the spectrum, lying between regular Inception modules and depthwise separable convolutions, hold further advantages.

![accuracy](https://user-images.githubusercontent.com/90513931/214254715-f525624f-59e1-418e-a271-e9ec2b24f30e.png)

![gradient](https://user-images.githubusercontent.com/90513931/214254724-10846f2e-6589-452c-b6b9-5e0f1caaee08.png)


