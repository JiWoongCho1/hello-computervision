## CBAM


To enhance performance of CNNs, recent researches have mainly investigated three important factors of networks. _depth, wideth, cardinality_. Apart from  these factors, they investigate a different aspect of the architecture design, _attention_. Attention not only tells where to focus, it also improves the representation of interests focusing on important features and suppressing unnecessary ones. In this paper, they propose a new network module, named "Convolutional Block Attention Module". Since convolution operations extract informative features by blending cross-channel and spatial information together, they adopt their module to emphasize meaningful features along those two principal dimensions: channel, spatial axis. To achieve this, they sequentially apply channel and spatial attention modules, so that each of the braches can learn 'what' and 'where' to attend in the channel and spatial axis respectively.

![module](https://user-images.githubusercontent.com/90513931/217157194-201c03a2-ecf8-4bd7-9a0e-ec4453021ce9.png)


#### Channel attention module

They produce a channel attention map by exploiting the inter-channel relationshup of features. As each channel of a feature map is considered as a feature detector, channel attention focuses on 'what' is meaningful given an input image. Beyond the previous works, they argue that max-pooling gathers another important clue about distinctive object features to infer finer channel-wise attention. Thus. they use both average pooled and max pooled features simultaneously. They compare 3 variantes of channel attention: average pooling, max pooling, and joint use of both poolings. (the channel attention module with an average pooling is the same as the SE module) Thus they suggest to use both features simultaneously and apply a shared network to those features.

![comparison_pooling](https://user-images.githubusercontent.com/90513931/217157206-2c86bf79-cb17-41d5-8fd1-62d52c98315c.png)


#### Spatial attention module

They generate a spatial attention map by utilizing the inter-spatial relationship of features. Different from the channel attention, the spatial attention focuses on 'where' is an informative part, which is complementary to the channel attenetion. To compute the spatial attention, they first apply average-pooling and max-pooling operations along the channel axis and concatenate them to generate an efficient feature descriptor. On the concatenated feature descriptor, they apply a convolution layer to generate a spatial attention map which encodes where to emphasize or suppress. To generate a 2D spatial attention map, they first compute a 2D descriptor that encodes channel information at each pixel over all spatial locations. They then apply on convolution layer to the 2D descriptor, obtaining the raw attention map. The final attention map is normalized by the sigmoid function. In the comparison of different convolution kernel sizes, they find that adopiting a larger kernel size generates better accuraccy in both cases. It implies that a broad view(large receptive field) is needed for deciding spatially important regions.

![comparison_kernel](https://user-images.githubusercontent.com/90513931/217157204-df0aff3e-e75e-49eb-827e-1679b7cf955b.png)

![channel_spatial_module](https://user-images.githubusercontent.com/90513931/217157201-5c4ef2c7-2b16-408c-aad9-8eb664bee5b2.png)


#### Arrangement of attention modules

Given an input image, two attention modules, channel and spatial, compute complementary attention, focusing on in a parallel or sequential manner. They found that the sequential arragement gives a better result than a parallel arrangement.


This table summarizes the experimental results. The networks with CBAM outperform all the baselines significantly, demonstraing that the CBAM can generalize well on various models in the large-scale dataset. Moreover, the models with CBAM improve the accuracy upon the one of the strongest methosd (SE module). It implies that their proposed approach is powerful, showing the efficacy of new pooling method that generates richer descriptor and spatial attention that complements the channel attention effectively. 

![comparison_error](https://user-images.githubusercontent.com/90513931/217157202-da803a79-826a-47be-ae4a-62ec76f3f9e1.png)

They also find that the overall overhead of CBAM is quite small in terms of both parameters and computation. This motivates them to apply their proposed module CBAM to the light-weight network, MobileNet.


![applying_mobilenet](https://user-images.githubusercontent.com/90513931/217157197-611b8cab-2cd5-4889-9bca-94c74dc4a6b9.png)


They conduct object detection on the MS COCO dataset. They adopt Faster-RCNN as their detecton method. And they are interested in performance imporovement by plugging CBAM to the baseline networks. 

![detection](https://user-images.githubusercontent.com/90513931/217157207-aadae0ce-8203-4347-8571-646d5c2478f8.png)
