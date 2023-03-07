## SSD(Single Shot Detector)

Faster R-CNN, operates at only 7 frames per second(FPS). There have been many attemps to build faster detectors by attacking each stage of the detection pipeline, but so far, significantly increased speed comes only at the cost of significantly decreased detection accuracy.

This paper presents the first deep network based object detector that does not resample pixels or features for bounding box hypotheses and is as accurate as approaches that do. This results in a significant improvement in sepeed for high accuracy  detection(59 FPS with mAP 74.3% on VOC2007 vs Faster R-CNN 7 FPS with mAP 73.2% or YOLO 45 FPS with mAP 63.4%)




#### Model

The SSD approach is based on a feed-forward convolutional network that produces a fixed size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections. The early network layers are based on a standard architecture used for high quality image classification, which authors call the base network. They add convolutional feature layers to the end of the truncated base network. These layers decrease in size progressively and allow predictions of detections at multiple scales. For a feature layer of sie m x n x with _p_ channels, the basice element for predicting parameters of a potential detection is a 3x3xp _small kernel_ that produces either a score for a category,, or a shape offset relative to the default box coordinates. 

They associate a set of default bounding boxes with  each feature map cell, for multiple feature maps at the top of the network. The default boxes tile the feautre map in a convolutional manner, so that the position of each box relative to its corresponding cell is fixed. At each feature map cell, they predict the offests relative to the default box shapes in the cell, was well as the per-class scores that indicate the presence of a class instance in each of these boxes. Speicifically, for each box out of _k_ at a given location, they compute _c_ class scores and the 4 offfsets relative to the original default box shape. Their default boxes are similar to the _anchor boxes_ used in Faster R-CNN, however they appley them to several feature maps of different resolutions. Allowing different default box shapes in several feature maps let us efficiently discretize the space of possible output box shapes. 


![SSD framework](https://user-images.githubusercontent.com/90513931/223299960-3610e948-0471-4a0a-9adb-5699f734997a.png)


![network comparison](https://user-images.githubusercontent.com/90513931/223299967-668da981-5a9b-420f-83af-1b4d2b1a40a9.png)


#### Training

During training they need to determine which default boxes correspont to a ground truth detection and train the network accordingly. For each ground truth box they are selecting from default boxes that vary over location, aspect ratio, and scale. They begin by matching each ground truth box to the default box with the bast jaccard overlap(IoU). They then match default boxes to any ground truth with jaccard overlap higher than a threshold(0.5). This simplifies the learning problem, allowing the network to predict high scores for multiple overlapping default boxes rather than requiring it to pick only the one with maximum  overlap. The overall objective loss function is a weighted sum of the localization loss and the confiden loss. The confidence loss is the softmax loss over multiple classes confidences.

![loss](https://user-images.githubusercontent.com/90513931/223305936-7231a649-1d72-428f-9b1c-f646caef96e2.png)

Previous works have shown that using feature maps from the lower layers can improve semantic segmentation quality because the lower layers capture more fine details of the input objects. Similarly, adding global context pooled from a feature map can help smooth the segmentation results. They use both the lower and upper feature maps for detection. In practice, they can use many more with small computational overhead.

Feature maps from different levels within a network are known to have different receptive field sizes. Fortunately, within the SSD framework, the default boxes do not necessary need to correspond to the actual receptive fields of each layer. They design the tiling of default boxes so that specific feature maps learn to be responsive to particular scales of the objects. After the matching step, most of the default boxes are negatives, especially when the number of possible default boxes is large. This introduces a significant imbalance between the positive and negative training examples. Instead of using all the negative examples, they sort them using the highest confidenceloss for eachc default box and pick the top ones so that the ratio between the negatives and positives is at most 3:1. They found that this leads to faster optimization and a more stable training.


#### Results

To understand the lerfr\ormance of SSD model in more detail, they note that SSD can detect various object categories with high quality. Compared to R-CNN, SSD has less localization error, indicating that SSD can localize objects better because it directly learns to regress the object shape and classify object categories instead of using two decoupled steps. However, SSD has more confusions with similar object categories, partly because authors share locations for multiple categories. Also they find that data augmentation and more default bounding box is crutial.

![result1](https://user-images.githubusercontent.com/90513931/223309368-058c1813-c7c7-440c-b842-c38ec7badba7.png)


![result2](https://user-images.githubusercontent.com/90513931/223309370-94068bd7-4525-4541-8b38-457d0015b2e2.png)
