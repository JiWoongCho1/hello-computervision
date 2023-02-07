## EfficientNet

Scaling up ConvNets is widely used to achieve better accuracy. For example, resNet can be scaled up from ResNet=18 to ResNet-200 by using more layers. However, the process of scaling up ConvNets has never been weel understood and there are currently many ways to do it. The most common way is to scale up ConvNets by their depth, width. Another less common,, but increasingly popular, method is to scale up models by image resolution. In previous work, it is common to scale only one of the three dimensions. In this paper, they want to stydy and rethink the process of scaling up ConvNets. In particular, they investigate the central question: is there a principled method to scale up ConvNets that can achieve bettter accuracy and efficiency? Their empirical stydy shows that it is critical to balance all dimensions of network width/depth/resolution, and surprisingly such balance can be achieved by simply scaling each of them with constant ratio. Based on this observation, they propsea simple yet effective _compound scaling method_.

Unlike regular ConvNet designs that mostly focus on finding the best layer architecture(F), model scaling tries to exoand the network length, width and resolution without changing F. By fixing F, model scaling simplifies the design problem for new resource constraints, but it still remains a large design space to explore different hyperparameters(width, depth, resolution) for each layer. In order to further reduce the design space, they restrict that all layers must be scaled uniformly with constant ratio.

#### Depth

Scaling network depth is the most common way used by many ConvNets. THe intuition is that deeper Net ccan capture richer and more complex features and generlaize well on new tasks. However, deeper networks are also more difficult to train due to the vanishing gradient problem. Although several techniques, such as skip conncections and batch normalization, alleviate the training problem, the accuracy gain of very deep network diminishes. For example, ResNet-1000 has similar acccuracy as ResNet-101 even though it has much more layers. 

#### Width

Scaling network width is commonly used for small size models. As discussed in wider networks tent to be able to capture more fine-grained features and are easier to train. However extremely wide but shallow networks tend to have difficulties in capturing higher level features.

#### Resolution

With higher resolution input images, Nets can potentially capture more fine-grained patterens. Starting from 224x224 in early, modern Nets tend to use 299x299 for better accuracy. Recently, GPipe achieves state-of-the-art accuracy with 480x480 resolution.

These analyses lead the scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models.

They empirically observe that different scaling dimensions are not independent. Intuitively, for higher resolution images, they should increase network depth, such that the larger receptive fields can help capture similar features that include more pixels in bigger images. Correspondingly, they should also increase network width when resolution is higher, in order to capture more fine-grained patterens with more pixels in high resolution imges. These intuitions suggest that they need to coordinate and balance different scaling dimensions rather than conventional single dimension scaling. To validate this intuitions, they compare these options.

These results lead that it is critical to balance all dimensions of network width, depth, and resolution during Network scaling.

So they propose a new compound scaling method, which use a compound corffcient θ to unformly scales network width, depth, resolution in a principled way, where alpha, beta, gamma are constants that can be determined by a small grid search.
