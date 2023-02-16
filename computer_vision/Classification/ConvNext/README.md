## ConvNext


CNNs have several built-in inductive biases that make them well suited to a wide variety of computer vision applications. Themost  important one is translation equivariance, which is a desirable property for tasks like objection detection. In the tear 2020, as the introduction of Vision Transformers completely altered the landscape of network architecture design. Vit introduce no image speicifc inductive bias and makes minimal changes to the original NLP Transformers. One primary focus of Vit is on the scaling behavior: with the help of larger model and dataset sizes, Transformers can outperform standard ResNets by a significant margin. Without the CNNs inductive bias, a vanila Vit model faces many challenges in begin adopted as a gegneric vision backbone. The biggest challenge is ViT's global attention design, which has a quadratic complexity with respect to the input size. This might be acceptable for ImageNet classficiation, but quickly becomes intractable with higher-resolution inputs. The 'sliding window' strategy was reintroduced to Transformers, allowing them to behave more similarily to CNNs. Swin Transformer is a milestone work in this direction, demonstrating for the first time that Transformers can be adopted as a generic vision backbone and achieve state-of-the-art performance across a range of computer vision tasks beyond image classification. Swin Transformer's success and rapid adoption also revealed one thing: the essence of convolution is not becoming irrelevantl rather, it remains much desired and has never faded. 

A naive implementation of sliding window self-attention can be expensive; with advanced approaches such as cycic shifting, the speed can be optimized but the system becomes more sophisticated in design. The only reason CNNs appear to be losing steam is that Transformers surpass them in many vision tasks, and the performance difference is usuallyattributed to the superior scaling behavior of Transformers, with multi head self attention being the key component. In this work, they investigate the architectural distinctions between CNNs and Transformers and try to identify the confounding variables when comparing the network performance. Their exploration is directed by a key question. "How do design decisions in Transforers impact CNNs performance?"

To comparing performances, they set the starting point is a ResNet-50. They then study a series of design decision which they summarized as 1) macro design, 2) ResNext, 3) inverted bottleneck, 4) large kernel size, and 5) various layer wise migcro designs. Apart from the design of network architectures, the training proceduce also affects the ultimate performance. Recent studies demonstrate that a set of modern ntraining techniques can significantly enhance the performance of a simple ResNet-50. By itself, thos enhaced trainign recipe increased the performance of the ResNet-50 model frm 76.1% to 78.8%, implying that a significant portion of the performance difference between tranditional CNNs and ViT may be due to the training techniques.

![comparison model](https://user-images.githubusercontent.com/90513931/219283848-b3d6897a-501a-47c7-a427-c83752c524a2.png)



#### Macro Design

The heavy 'res4' stage was meant to be compatiable with downstream tasks like object detection, where a detecor head operates on the 14 x 14 feature plane. Swin-T, on the other hand, followed the same principle but with a slightly different stage compute ratio of 1:1:3:1. For larger Swin Transformers the ratio is 1:1:9:1. Following the design, they adjust the number of the blocks in each stage from (3,4,6,3) in ResNet-50 to (3,3,9,3), which also aligns the FLOPs with Swin-T. This improves the model accuracy from 78.8% to 79.4%. The stem cell in standard ResNet contains a 7x7 convolution layer with stride 2, followed by a max pool, which results in a 4x downsampling of the input images. In vision-Transformers a more aggressive 'patchify' strategy is used as the stem cell, which corresponds to a large kernel size and non overlapping convolution. They replace the ResBet-style stem cell with a patchify layer implemented using a 4x4, stride 4 convolutional layer. The accuracy has chaned from 79.4$ to 79.5%. 



#### ResNeXt-ify

ResNeXt model has a better FLOPs/accuracy trade-off than a vanila ResNet. THe core component is grouped convolution, where the convolutional filters are separated into different groups. More precisely, ResNeXt employs grouped convolution for the 3x3 conv layer in a bottleneck block. The combination of depthwise conv and 1x1 convs leads to a separation of spatial and channel mixing, property shared by vision Transformers, where each operation either mixes information across spatial or channel dimension, but not both. The use of depthwise convolution efectively reduces the network FLOPs and, as expected, the accuracy. THis brings the network performance to 80.5% with increased FLOPs.

#### Inverted Bottleneck

This Transformer design is connected to the inverted bottleneck design with an expansion ratio of 4 used in CNNs.  The idea was popularizezd by MobileNetV2, and has subsequently gained traction in several advanced CNNs architectures. They explore the inverted bottleneck design. Despite the increased the FLOPs for the depthwise convolution layer, this change reduces the whole network FLOPs to 4.6G, due to the significant FLOPs reduction in the downsampling residual blocks' shortcut 1x1 convlayer. This results in slightly imporved performance from 80.5% to 80.6%.



#### Large Kernel Sizes

One of the most distinguishing aspects of vision Transformer is their non-local self-attention, which enables each layer to have a global receptive field. To explore large kernels, one prerequistite is to move up the position of the depthwise conv layer.The complex/inefficient modules(MSA, large kernel conv) will have fewer channels, while the efficient, dense 1x1 layers will do the heavy lifting. This intermediate step deduces the FLOPs to 4.1G, resulting in a temporary performance degradation to 79.9%. And they experimented with several kernel sizes, including 3,5,7,9,11. The networks performance increases from 79.9%(3x3)to 80.6%(7x7). while the network's FLOPs stay roughtly the same. 



#### Micro Design

Rectified Linear Unit(ReLU) is still extensively used in CNNs due to its simplicity and efficieny. ReLU is also used as an activation function in the original Transformer paper. The Gaussian Error Linear Unit(GELU), which can be thought of as a smoother variant of ReLU. So they find the ReLU can be subsituted with GELU in CNNs too, although the accuracy stays unchanged. 

One minor distinctin between a Transformer and a Resnet block is that Transformers have fewer activation functions. So they examine how performance changes when they stick to the same strategy. They eliminate all GELU layers from the residual block except for one between two 1x1 layers, replicating the style of a Transformer block. This process improves the result by 0.7% to 81.3%, pratically matching the performance of Swin-T. 

![applying technique](https://user-images.githubusercontent.com/90513931/219283556-18104a8b-9b8c-43b8-b253-e32212b5d305.png)

Transformer blocks usually have fewer normailzation layers as well. Here they remove two BatchNorm layers, leaving only on BN layer before the conv 1x1 layers. As empirically they find that adding one additional BN layer at the beginning of the block doew not improve the performance. 

BatchNorm is an essential component in CNNs as it improves the convergence and reduces overfitting. However, BN also has many intricacies that can have a detrimental effect on the model's performance. On the other hand, the simpler Layer Normalization has been used in  Transformers, resulting in good performance across different application scenarios. Directly substituting LN for BN in the original ResNet will result in suboptimal performance. The performance is slightly better, obtaining an accuracy of 81.5%. 

In Swin Transformers, a separate downsampling layer is added between stages. They explore a similar strrategy in which they use 2x2 con layers with stride 2 for spatial downsampling. This modification surprisingly leads to diverged training. Further investigation shows that, adding normalization layes wherevery spatial resolution is changed can help stablize training. THese include several LN layers also used in Swin Transformers. They can improve the accuracy to 82%, significantly exceeding Swin-T's 81.3%


This figure shows the comparison to the models. Without specialized modules such as shifted windows or relative position bias, ConvNeXt also enjoy improved throughput compared to Swin Transformers.

![classification accuracy](https://user-images.githubusercontent.com/90513931/219283558-248491f8-2a5b-4a06-939d-e9c7f207cd4a.png)


They finetune Mask R-CNN and cascade Mask R-CNN on the COCO dataset with ConvNeXt backbones. Next figure shows object detection and instance segmentation results comparing Swin Transformer, ConvNeXt, and tranditional CNNs such as REsNeXt. We can note that ConvNeXt achieves on-par or better performance than Swin Transformer.

![detection accuracy](https://user-images.githubusercontent.com/90513931/219283553-ac2cbaec-9e83-4e16-8a0a-c5af63e33a3f.png)


In comparision to vanila ViT, both ConvNeXt and Swin Transformer exhibit a more favorable accuracy FLOPs trade-off due to the local computations. It is worth noting that this improved efficiency is a result of the _CNNs inductive bias_, and is not directly related to the self-attention mechanism in vision Transformers.

