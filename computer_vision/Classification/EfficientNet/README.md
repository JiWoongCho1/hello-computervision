## EfficientNet

Scaling up ConvNets is widely used to achieve better accuracy. For example, ResNet can be scaled up from ResNet=18 to ResNet-200 by using more layers. However, the process of scaling up ConvNets has never been well understood and there are currently many ways to do it. The most common way is to scale up ConvNets by their depth, width. Another less common, but increasingly popular, method is to scale up models by image resolution. In previous work, it is common to scale only one of the three dimensions. In this paper, they want to study and rethink the process of scaling up ConvNets. In particular, they investigate the central question: is there a principled method to scale up ConvNets that can achieve bettter accuracy and efficiency? Their empirical study shows that it is critical to balance all dimensions of network width/depth/resolution, and surprisingly such balance can be achieved by simply scaling each of them with constant ratio. Based on this observation, they propose simple yet effective _compound scaling method_.

![compound_scaling](https://user-images.githubusercontent.com/90513931/217189107-c4618590-dc3c-4b5a-b1f4-f7a50944ca34.png)


Unlike regular ConvNet designs that mostly focus on finding the best layer architecture(F), model scaling tries to expand the network length, width and resolution without changing F. By fixing F, model scaling simplifies the design problem for new resource constraints, but it still remains a large design space to explore different hyperparameters(width, depth, resolution) for each layer. In order to further reduce the design space, they restrict that all layers must be scaled uniformly with constant ratio.

#### Depth

Scaling network depth is the most common way used by many ConvNets. The intuition is that deeper Net can capture richer and more complex features and generlaize well on new tasks. However, deeper networks are also more difficult to train due to the vanishing gradient problem. Although several techniques, such as skip conncections and batch normalization, alleviate the training problem, the accuracy gain of very deep network diminishes. For example, ResNet-1000 has similar acccuracy as ResNet-101 even though it has much more layers. 

#### Width

Scaling network width is commonly used for small size models. As discussed in wider networks tend to be able to capture more fine-grained features and are easier to train. However extremely wide but shallow networks tend to have difficulties in capturing higher level features.

#### Resolution

With higher resolution input images, Nets can potentially capture more fine-grained patterens. Starting from 224x224 in early, modern Nets tend to use 299x299 for better accuracy. Recently, GPipe achieves state-of-the-art accuracy with 480x480 resolution.

These analyses lead the scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models.

![FLOPS each dimension](https://user-images.githubusercontent.com/90513931/217189110-97e73bf9-cb02-4a9e-a207-cb2af61919d0.png)

They empirically observe that different scaling dimensions are not independent. Intuitively, for higher resolution images, they should increase network depth, such that the larger receptive fields can help capture similar features that include more pixels in bigger images. Correspondingly, they should also increase network width when resolution is higher, in order to capture more fine-grained patterens with more pixels in high resolution imges. These intuitions suggest that they need to coordinate and balance different scaling dimensions rather than conventional single dimension scaling. To validate this intuitions, they compare these options.

These results lead that it is critical to balance all dimensions of network width, depth, and resolution during Network scaling.

![table_compound](https://user-images.githubusercontent.com/90513931/217189116-de3cc3ee-0ba2-445e-bf1a-12145834aa4e.png)

So they propose a new compound scaling method, which use a compound corffcient θ to unformly scales network width, depth, resolution in a principled way, where alpha, beta, gamma are constants that can be determined by a small grid search.

![compound method](https://user-images.githubusercontent.com/90513931/217189126-4f45b1cc-a8a3-48ce-911b-44294ab8a962.png)


They evaluate their scaling method using existing Nets, but in order to better demonstrate the effectiveness of their scaling method, they have also developed a new mobile size baseline, called EfficientNet. Since they use the same search space, the architecture is similar to MnasNet, except their EfficientNet-B0 is slightly bigger due to the larger FLOPS target. Notably, it is possible to achieve even better performance by searching for alpha, beta, gamma directly around a large model, but the search cost becomes prohibitively more expensive on larget models.

![baseline network](https://user-images.githubusercontent.com/90513931/217189121-77f2c348-ac55-461b-8645-8d615f991c95.png)

To disentangle the contribution of their proposed scaling method from the EfficientNet architecture, this table compares the performance of different scaling methods for the same baseline network. In general, all scaling methods improve accuracy with the cost of more FLOPS, but their compound scaling method can further improve accuracy, by up to 2.5%, than other single dimension scaling methods, suggesting the importance of their proposed compound scaling.


![comparing acc](https://user-images.githubusercontent.com/90513931/217189124-64a1f0fa-33d0-4341-bea3-0611a470988e.png)


![model performance](https://user-images.githubusercontent.com/90513931/217189112-9de131c8-08dd-4dd7-bb18-b1b88682323e.png)
