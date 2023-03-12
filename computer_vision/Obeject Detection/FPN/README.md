## FPN(Feature Pyramid Networks)


Aside from being capable of representing higher-level semantics, ConvNets are also more robust to variance in scale and thus facilitate recognition from features computed on a single input scale. But even with this robustness, pyramids are still needed to get the most accurate results. The principal advantage of featurizing each level of an image pyramid is that it produces a multi-scale feature representation in which _all levels are semantically strong_, including the high-resolution levels.

Nevertheless, featurizing each level of an image pyramid has obvious limitations. Inference time increases considerably, making this approach impractical for real applications. Moreover, training deep networks end-to-end on an image pyramid is infeasible in terms of memory, and so, if exploited, image pyramids are used only at test time, which creates an inconsistency between train/test-time inference. However, image pyramids are not the only wy to compute a multi-scale feature representation. A deep ConvNet computes a _feature hierarchy_ layer by layer, and with subsampling layers the feature hierarchy has an inherenct multi scale, pyramidal shape. This is in-network feature hierarchy produces feature maps of different spatial resolutions, but introduces large semantic gaps caused by different depths.

The Single Shot Detector(SSD) is one of the first attempts at using a ConvNet's pyramidal. Ideally, the SSD-style pyramid would reuse the multi-scale feature maps from different layers computed in the forward pass and thus come free of cost. But to avoid using low-level  features SSD forgoes reusing already computed layers and instead builds the pyramid starting from high up in the network and then by adding several new layers. Thus it misses the opportunity to reuse the higher-resolution maps of the feature hierarchy. So authors show that these are important for detecting small objects. 

The goal of this paper is to naturally leverage the pyramidal shape of a ConvNet's feature hierarchy while creating a feature pyramid that has strong semeantics at all scales. To achieve this goal, they raly on an architecture that combines low-resolution, semantically strong fetaures with high-resolution, semantically weak features via a top-down pathway and lateral connections. The result is a feature pyramid that has rich semantics at all levels and is build quickly from a single input image scale. They called a Feature Pyramid Network(FPN).

(figure 1 사진)


#### Bottom-up pathway

The bottom-up pathway is the feed-forward computation of the backbone ConvNet, which computes a feature hierarchy consisting of feature maps at several scales with a scaling step of 2. There are often many layers producing output maps of the same size and they say these layers are in the same network _stage_. They choose the output of the last layer of each stage as their reference set of feature maps, which they will enrich to create their pyramid. This choice is natrual since the deepest layer of each stage should have the strongest features.

#### Top-down pathway and lateral connections

The top-down pathway hallucinates higher resolution features by upsampling spatially coarser, but semantically stronger, feature maps from higher pyramid levels. These features are then enhanced with features from the bottom-up pathway via lateral connections(skip connections). Each lateral connection merges feature maps of the same spatial size from the bottom-up pathway and the top-down pathway. The bottom-up feature map is of lower-level semantics, but its activation are more accurately localized as it was subsampled fewer times. With a coarser-resolution feature map, they upsample the spatial resoluton by a factor of 2(using nearest neighbor upsampling for simplicity). The upsampled map is then merged with the corresponding bottom-up map(which undergoes a 1x1 convolutional layer to reduce channel dimensions) by element-wise addition. This process is iterated until the finest resolution map is generated. Simplicity is central to this design and they have found that the model is robust to many design choices.

(figure 3 사진)



#### RPN

They adopt their method in RPN for bounding box proposal generation and in Fast-R-CNN for object detection. They adapt RPN by replacing the single-scale feature map with their FPN. They attach a head of the same design(3x3 conv and two sibling 1x1 convs) to each level on their feature pyramid. Because the heads slides densely over all locations in all pyramid levels, it it not necessary to have multi-scale anhors on a specific level. Instead, they assign anchors of a single scale to each level. Formally, they define the anchors to have areas of {32^2, 64^2, 128^2, 256^2, 512^2}. They note that the parameters of the heads are shared across all feature pyramid levels" they have also evalutated the alternative without sharing parameters and observed similar accuracy. The good performance of sharing parameters indicates that all levels of their pyramid shre similar semantic levels. this advantage is analogous to that of using a featurized image pyramid, where a common head classifier can be applied to features computed at any image sale.

#### Feature Pyramid Networks for Fast R-CNN

Fast R-CNN is most commonly performed on a single-scale feature map. To use it with the FPN, they need to assign RoIs of different scales to the pyramid levels. Thus they can adapt the assignment strategy of region-based detectors in the case when they are run on image pyramids. They attach predictor heads to all RoIs of all levels. They simply adopt RoI pooling to extract 7x7 features, and attach two hidden 1024-d fully connected layers before the final classification and bounding box regression layers. Based on these adaptations, they can train and test FastR-CNN on top to the feature pyramid.
