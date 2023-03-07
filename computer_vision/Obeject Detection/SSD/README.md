## SSD(Single Shot Detector)

Faster R-CNN, operates at only 7 frames per second(FPS). There have been many attemps to build faster detectors by attacking each stage of the detection pipeline, but so far, significantly increased speed comes only at the cost of significantly decreased detection accuracy.

This paper presents the first deep network based object detector that does not resample pixels or features for bounding box hypotheses and is as accurate as approaches that do. This results in a significant improvement in sepeed for high accuracy  detection(59 FPS with mAP 74.3% on VOC2007 vs Faster R-CNN 7 FPS with mAP 73.2% or YOLO 45 FPS with mAP 63.4%)

(Figure 1사진)

