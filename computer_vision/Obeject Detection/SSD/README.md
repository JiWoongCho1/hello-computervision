## SSD(Single Shot Detector)

Faster R-CNN, operates at only 7 frames per second(FPS). There have been many attemps to build faster detectors by attacking each stage of the detection pipeline, but so far, significantly increased speed comes only at the cost of significantly decreased detection accuracy.

This paper presents the first deep network based object detector that does not resample pixels or features for bounding box hypotheses and is as accurate as approaches that do. This results in a significant improvement in sepeed for high accuracy  detection(59 FPS with mAP 74.3% on VOC2007 vs Faster R-CNN 7 FPS with mAP 73.2% or YOLO 45 FPS with mAP 63.4%)

![SSD framework](https://user-images.githubusercontent.com/90513931/223299960-3610e948-0471-4a0a-9adb-5699f734997a.png)


#### Model
The SSD approach is based on a feed-forward convolutional network that produces a fixed size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections. The early network layers are based on a standard architecture used for high quality image classification, which authors call the base network. They add convolutional feature layers to the end of the truncated base network. These layers decrease in size progressively and allow predictions of detections at multiple scales. For a feature layer of sie m x n x with _p_ channels, the basice element for predicting parameters of a potential detection is a 3x3xp _small kernel_ that produces either a score for a category,, or a shape offset relative to the default box coordinates.


![network comparison](https://user-images.githubusercontent.com/90513931/223299967-668da981-5a9b-420f-83af-1b4d2b1a40a9.png)
