## BAM(Bottleneck Attention Module)

A fundamental approach to boost performance is to design a good backbone archcitecture. The most intuitive way to boost the network performance is to stack more layers. Deep neural networks then are able to approximate high dimensional function using their deep layers. Apart from the tranditional approaches, authors investigate the effect of _attention_ in DNNs, and propose a simple, light-weight module for general DNNs. That is, the propsed module is designed for easy integration with existing CNN architectures. While most of the previous works use attention with task-specific purposes, they explicitly investigate the use of attention as a way to imporve network's representation power in an extremely efficient way. As a result they propose "Bottleneck Attention Module" a simple and efficient attention module that can be used in any CNNs. As the channels of feature maps can be regarded as feature detectors, the two branches explicitly learn 'what' and 'where' to focus on.


![architecture](https://user-images.githubusercontent.com/90513931/216559212-e19bd48a-1717-4623-9b8f-2336ffa2f623.png)


In order to efficiently and adpatively process visual information, human visual systems iteratively process spatial glimpses and focus on salient areas. 'Squeeze and Excitation' module is also similar but it misses the spatial axis, which is important factor in inferring accurate attention map. 


#### Channel attention branch

As each channel contains a specific feature response, they exploit the inter-channel relationshup in the channel branch. To aggregate the feature map in each channel, they take global average pooling on the feature map and produce a channel vector(C x 1 x 1). This vector softly encodes global information in each channel. To estimate attention across channels form the channel vector, they use a multi layer perceptron with one hidden layer.

#### Spatial attention branch

The spatial branch produces a spatial attention map to emphasize or suppress features in different spatial locations. It is widely known that utilizing contextual information is crutial to know which spatial locations should be focused on. It is important to have a large receptive field to effectively leverage contextual information. So they employ dilated convolution to enlarge the receptive fields with high efficiency.

#### Combine two attention branches.

After acquiring the channel attention and the spatial attention from two attention branches, they combine them to produce their final 3D attention map. Since the two attention maps have different shrapes, they select, apply element wise summation for efficient gradient flow. After the summation, they take a sigmoid function to obtain the final 3D attention map in the range from 0 to 1. This 3D attention map is element wisely multiplied with the input feature map then is added upon the original input feature map to acquire the refined feature map.

![attention module](https://user-images.githubusercontent.com/90513931/216559216-36fd8658-4624-45ee-8a41-5de11a110a0e.png)


During construct module, select hyperparameters, they conduct the ablation study to validate their design choice in the module. And this table shows that combining the channel and spatial branches together play a critical role in inferring the final attention map. In this experiment, they emprirically verify that the significant improvement does not come from the increased depth by naviely adding the extra layers to the bottlenecks. They can obviously notice that plugging BAM not only produces superior performance but also puts less overhead than naively placing the extra layers. It implies that the improvement of BAM is not merely due to the increased depth but because of the effective feature refinement.

![experiment](https://user-images.githubusercontent.com/90513931/216559204-de11978f-c55e-4c9c-9839-6b172a5846ca.png)

And they also verify they bottlenecks of networks are effective points to place their module BAM. Recent studies on attention mechanisms mainly focus on modifications within the 'convolutional blocks' rather than the 'bottlenecks'. They compare and observe that placing the module at the bottleneck is effective in terms of overheadd/accuracy trade-offs. 

They conduct object detection on MS COCO dataset. They adopt Faster-RCNN as their detection method and ImageNet pre-trained REsNet101 as a baseline network, observing significant improvements from  the baseline, demonstrating generalization performance of BAM on other recognition tasks.


They conduct additional experiments to compare their method with SE. Their module requires slightly more GFLOPS but has much less parameters than SE, as they place their module only at the bottlenecks not every conv blocks.

![comparing with SE](https://user-images.githubusercontent.com/90513931/216559218-6937bff0-4227-446e-9427-73a3c1111f87.png)
