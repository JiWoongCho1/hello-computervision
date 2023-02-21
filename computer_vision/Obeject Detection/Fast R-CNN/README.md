## Fast R-CNN


Compared to image classification, object detection is a more challenging task that requires more complex methods to solve. Complexity arises because detection requires the accurate localization of objects, creating two primary challenges. First, numerous candidate object locations(region proposals) must be processed. Second, these candidates provide only rough localization that must be refined to achieve precise localization. Solutions to these problems often compromise speed, accuracy, or simplicity. In this paper, they streamline the training process for state-of-the-art ConvNet-based object detectors. They propose a single stage training algorithm that jointly learns to classiffy object proposals and refine their spatial locations. 

The R-CNN achieves excellent object detection accuracy by using a deep ConvNet to classify object proposals. R-CNN, however, has notable drawbacks: 1) Training is a multi stage pipeline. 2) Training is expensive in space and time. 3) Object detection is slow.

So they propose a new training algorithm that fixes the disadvantages of R-CNN and SPPnet, while improving on their speed and accuracy. They call this method _Fast R-CNN_ because it's comparatively fast to traind and test. The Fast R-CNN method has several advantages. 1) Higher detection quality(mAP) than R-CNN, SPPNet. 2) Training is single-stage, using a multi-task loss. 3) Training can update all network layers. 4) No disk storage is required for feature caching. 

This figure show the Fast R-CNN architecture. A fast R-CNN network takes as input an entire image and a set of object proposals. The network first processes the whole image with several convolutional and max pooling layers to produce a conv feature map. Then, for each object proposal a region of interest pooling layer extracts a fixed-length fetaure vector from the feature map. Each feature vector is fed into a sequentce of fully connected layers that finally branch into two sibling output layers: one that produces softmax probability over K object classes plus a background class and another layer is that four real valued numbers for each of the K object classes.

![architecture](https://user-images.githubusercontent.com/90513931/220299788-0d44c1bf-aabc-4810-b916-f9b8a0db26ad.png)

#### RoI pooling layer

The RoI pooling layer uses max pooling to convert the features inside any valid region of interst into a small feature map with a fixed spatial extent of H x W(7x7), where H and W are layer hyper parameters that are independent of any particular. RoI max pooling works by dividing the hxw RoI window into an HxW grid of subwindows of approximate size h/W x w/W and then max-pooling the values in each sub-window into the corresponding output grid cell. The RoI layer is simply the special case of the spatial pyramid pooling layer used in SPPnets in which there is only one pyramid level.

To fine-tuning for detection, they apply the multi-task loss, mini-batch sampling, backpropagation through RoI pooling layers, and SGD hyper parameters. 

The network takes as input an imageand a list of R object proposals to score. To test time, R is typically around 2000, although they will consider cases in which it is larger(45k). When using an image pyramid, each RoI os assigned to the scale such that the scaled RoI is closest to 224^2 ixels in area. 

For whole image classification, the time spent computing the fully connected layers is small compared the conv layers. On the contrary, for detectin the number of RoIs to process is large and nearly half of the forward pass time is spent computing the fully connected layers. Large fully connected layers are easily accelerated by compressing them with truncated SVD.

For VOC 2010 and 2012, Fast R-CNN achieves the top result on VOC 12 with a mAP of 65.7%. It is also two orders of magnitude faster than the other methods, which are all based on the 'slow' R-CNN pipeline. 

![VOC2007 comparison](https://user-images.githubusercontent.com/90513931/220299785-ebf543de-fa3f-427d-82a5-266dee76e03e.png)

#### SVM vs softmax

Fast R-CNN uses the softmax classifier learnt during fine-tuning instead of training one-vs-rest linear SVMs post-hoc, as was done in R-CNN and SPPnet. To understand the impact of this choice, they implemented post-hoc SVM training with hard negative mining in Fast R-CNN. This table shows softmax slightly outperforming SVM for all three networks, by +0.1 to +0.8 mAP points. 

![softmax vs SVM](https://user-images.githubusercontent.com/90513931/220299784-f7690ca6-fc6a-41f5-9485-53efa3f86652.png)

#### Are more proposals always better?

There are two types of object detectors: those that use a sparse set of object proposals and those that use a dense set. Classifiying sparse proposals is a type of _cascade_ in which the proposal mechanism first rejects a vast number of candidates leaving the classifier with a small set to evaluate. They find that mAP rises and then falls slightly as the proposal count increases. This experiment shows that swamping the deep classifier with more proposals does not help, and even slightly hurts, accuracy. 


![numbers of proposals](https://user-images.githubusercontent.com/90513931/220299772-5a33b169-71cd-4c85-b762-91190b05bd14.png)

