## R-CNN

The centreal issue can be distilled to the following: To what extent do the CNN classification results on ImageNet generalize to object detection results on the PASCAL VOC challenge? This paper is the first to show that a CNN can lead to dramatically higher object detection performance on PASCAL VOC as compared to systems based on simpler HOG-like features. To acheive this result, authors focused on two problems: localizing objects with a deep network and training a high-capacity model with only a small quantity of annotated detection data.


They solve CNN localization problem by operating within the "recognition using regions" paradigm, which has been successful for both object detection and semantic segmentation. And they use a simple technique(affine image warpig) to compute a fixed-size CNN input from each region proposal, regardless of the region's shape. This figure presents an overview of their method and highlights some of their results. Since their system combines region proposal with CNNs, they dub the method R-CNN, Region with CNN features.

![overview](https://user-images.githubusercontent.com/90513931/220125963-5dad1dce-6998-4339-937e-0d9dbfb0f8c8.png)


Next challenge faced in detection is that labeled data is scarece and the amount the currently available is insufficient for training a large CNN. The conventional solution to this problem is to use _unsupervised_ pre-training, followed by supervised fine tuning.The second principle contribution of this paper is to show that _supervised_ pre-training on a large auxiliary database(ILSVRC), followed by domain specifi fine-tuning on a small dataset(PASCAL), is an effective paradigm for learning high-capacity CNNs when data is scarece. This system is also quite efficient because the only class-specific computations are a reasonably small matrix-vetor product and greedy non maximum suppression. They note that because R-CNN operates on regions it is natural to extend it to the task of semantic segmentation. With minor modifications, they also acheive competitive results on the PASCAL VOC segmentation task, with an average segmentation accuracy of 47.9% on the VOC 2011 test set.

#### Obejct detection with R-CNN

Thier object detection system consist of three modules. The first generates category-independent region proposals. These proposals define the set of candidate detections available to their detector. The second module is a large convolutional netural network that extracts a fixed-length feature vector from each region. The third module is a set of class-specific linear SVMs.

#### module design
There has various method for region proposals, while R-CNN is agnostic to the particular region proposal method, so they use selective search to enable a controlled comparison with prior detection work. And they extract a 4086 dimensional feature vector from each region proposal using the Caffe implementation of the AlexNet. In order to compute features for a region proposal, they must first convert the image data in that region into a form that is compatible with the CNN.(227 x 227, because of network is AlexnNet)

#### Test time detection
At test time, they run selective search on the test image to extract around 2000 region proposals. They warp each proposal and forward propagate it through the CNN in order to compute features. Then, for each class, they score each extracted feature vector using SVM trained for that class. Given all scored regions in an image, they apply a greedy non-maximum suppression that rejects a region if it has asn intersection-over-union(Iou) overlap with a higher scoring selected region larger than a learned threshold. Two properties make detection efficient. First, all CNN parameters are shared across all categories. Second, the feature vectors computed by the CNN are low-dimensional when compared to other common approaches. This analysis shows that R-CNN can scale to thousands of object classes without resorting to approximate techniques, such as hashing. Even if there were 100k classes, the resulting amtrix multiplication takes only 10seconds on a modern multi-core CPU. This efficiency is not merely the result of using region proposals and shared features.


They now look at results from their CNN after having fine-tuned its parameters on VOC 2007 trainval. The improvement is striking: fine-tuning increases mAP by 8.0 percentage points to 54.2%. Most results in this paper use the network architecture from AlexNet. However, the have found that the choice of architecture has a large effect on R-CNN detection performance. 

![comparison](https://user-images.githubusercontent.com/90513931/220125974-8a2bc682-d0a6-4b2d-be2e-4f612baa21fe.png)
