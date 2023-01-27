## MobileNetV2

This paper introduced a new neural network architecture that is specifically tailored for mobile and resource constrained environments by decreasing the number of operations and memory needed while retaining the same accuracy. Their main contribution is a novel layer module: the inverted residual with linear bottleneck.

Recently there has been lots of progress in algorithmic architecture exploration included hyperparameter optimization as well as various methods of network pruning and connectivity learning. However one of the drawback is that the resulting networks end up very complex. Their network design is based on MobileNetV1 that is simplicity and does not require any special operators.

Depthwise Seperable Convolutions that is main idea and technique of the MobileNetV1 are a key building block for this architecture. The basic idea is to replace a full convolutional operator with a factorized version that splits convolutiona into two seperate layers.(depthwise, pointwise. It is well described in my 'MobileNetV1' folder.

#### Linear Bottlenecks

Informally, for an input aet of real images, they say that the set of layer activations forms a 'maniford of interest'. It has been long assumed that manifolds of interest in neural networks could be embedded in low-dimensional subspaces. At a first glance, such a fact could then be captured and exploited by simply reducing the dimensionality of a layer thus reducing the dimensionality of the operating space. Following this intuition, the width multiplier approach allow one to reduce the dimensionality of the activation space until the manifold of the interest spans this entire space. It is easy to see that in general if a result of a layer transformation 'ReLU' has a non-zero volume S, the points mapped to interior S are obtained via a linear transformation B of input, thus indicating that the part of the input space corresponding to the full dimensional output, is limited to a linear transformation. To summarize, ReLU is capable of preserving complete information about the input manifold, but only if the input manifold lies in a low-dimensionaly subspace of the input space. These insights provide the hints for optimizing existing neural architectures. Assuming the manifold of interest is low-dimensional they can capture this by inserting _linear bottleneck_ layers into the convolutional blocks. Experimental evidence suggests that using linear layers is crutial as it prevents non-linearities from destroying too much information.

#### Inverted residuals

Inspired by the intuition that the bottlenecks actually contain all the necessary informations, while an expansion layer acts merely as an implementation detail that accomapanies a non-linear transformation of the tensor, they use shortcus directly between the bottlenecks. The inverted design is considerably more memory efficiend as well as works slightly better in their experiments and this allows to use mobile applications. The importance of residual connection has been studied extensively. The new result reported in this paper is that the shortcuts connecting bottleneck perform better than shortcuts connecting the expanded layers. 

And they tried to set the 't' is the number of the expansion ratio, find it being a small constant 2 to 5.

They evaluate and compare the performace of MobileNetV2 and MobileNetV1 as feature extractors for object detection with a modified version of the SSD on COCO dataset. They introduced a mobile friendly variant of regular SSD. They replace all the regular convolutions with seperable convolutions in SSD prediction layers and they this calls SSDLite.
