## CBAM


To enhance performance of CNNs, recent researches have mainly investigated three important factors of networks. _depth, wideth, cardinality_. Apart from  these factors, they investigate a different aspect of the architecture design, _attention_. Attention not only telss where to focus, it also improves the representation of interests focusing on important features and suppressing unnecessary ones. In this paper, they propose a new network module, named "Convolutional Block Attention Module". Since convolution operations extract informative features by blending cross-channel and spatial information together, they adopt their module to emphasize meaningful features along those two principal dimensions: channel, spatial axis. To achieve this, they sequentially apply channel and spatial attention modules, so that each of the braches can learn 'what' and 'where' to attent in the channel and spatial axes respectively.

#### Channel attention module

They produce a channel attention map by exploiting the inter-channel relationshup of features. As aeach channel of a feature map is considered as a feature detector, channel attention focuses on 'what' is meaningful given an input image. Beyond the previous works, they argue that max-pooling gathers another important clue about distinctive object features to unfr=er finer channel-wise attention. Thus. they use both average pooled and max pooled features simultaneously. They compare 3 variantes of channel attention: average pooling, max pooling, and joint use of both poolings. (the channel attention module with an average pooling is the same as the SE module.) Thus they suggest to use both features simultaneously and apply a shared network to those features.

#### Spatial attention module

They generate a spatial attention map by utilizing the inter-spatial relationship of features. Different from the channel attention, the spatial attention focuses on 'where' is an informative part, which is complementary to the channel attenetion. To compute the spatial attention, they first apply average-pooling and max-pooling operations along the channel axis and concatenate them to generate an efficient feature descriptor. On the concatenated feature descriptor, they apply a convolution layer to generate a spatial attention map which encodes where to emphasize or suppress. To generate a 2D spatial attention map, they first compute a 2D descriptor that encodes channel information ar each pixel over all spatial locations. They then apply on convolution layer to the 2D descriptor, obtaining the raw attention map. The final attention map is normalized by the sigmoid function. In the comparison of different convolution kernel sizes, they find that adopiting a larger kernel size generates better accuraccy in both cases. It implies that a broad view(large receptive field) is needed for deciding spatially important regions.

#### Arrangement of attention modules

Given an input image, two attention modules, channel and spatial, compute complementary attention, focusing on in a parallel or sequential manner. They found that the sequential arragement gives a better result than a parallel arrangement.


This table summarizesthe experimental results. The networks with CBAM outperform all the baselines significantly, demonstraing that the CBAM can generalize well on various models in the large-scale dataset. Moreover, the models with CBAM improve the accuracy upon the one of the strongest methosd (SE module). It implies that their proposed approach is powerful, showing the efficacy of new pooling method that generates richer descriptor and spatial attention that complements the channel attention effectively. They also find that the overall overhead of CBAM is quite small in terms of both parameters and computation. This motivates them to apply their proposed module CBAM to the light-weight network, MobileNet.

They conduct object detection on the MS COCO dataset. They adopt Faster-RCNN as their detecton method. And they are interested in performance imporovement by plugging CBAM to the baseline networks. 
