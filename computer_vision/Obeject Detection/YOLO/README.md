## YOLO v1

They reframe object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities. Using this system, we only look once(YOLO) at an image to predict what objects are present and where they are. YOLO is refreshingly simple. Next figure show that single CNN simultaneously predicts multiple bounding boxes and class probabilities for those boxes. YOLO trains on full images and directly optimizes detection performance. This unified model has several benefits over tranditional methods of object detection. 

![CNN predict](https://user-images.githubusercontent.com/90513931/223003996-07b83dbe-58fb-451a-9bf7-a26f421d05cf.png)!


First, YOLO is extremly fast. Since authors frame detection as a regression problem they don't need a complex pipeline. Second, YOLO reasons globally about the image when making predictions. Unlike sliding window and region propoals based techneiques, YOLO sees the entire image during training and test time so it implicitly encodes contextual information about classes as well as their appearance. Third, YOLO learns generalizable representations of objects. When trained on natural images and tested on artwork, YOLO outperforms top detection methods like DPM and R-CNN by a wide margin.

#### Unified Detection

This network uses features from the entire image to predict each boundig box. It also predicts all bounding boxes across all classes for an image simultaneously. This YOLO design enables end-to-end training and real time speeds while maintaining high average precision. This system divides the input image into an S x S grid. If the center of an object falls into a grid cell, that grid cell is responsible for detection that object. Each grid cell predicts 'B' bounding boxes and confidence scores for those boxes. These confidence score reflect how confident the model is that box contains an object and also how accurate it thinks the box is that it predicts. Formally they define confidence as "Pr(object) x IOU(truth predict)". Each bounding box consists of 5 Predictions: x,y,w,h, and confidence. (x,y) coordinates represent the center of the box relative to the bounds of the grid cell. The width and height are predicted relative to the whole image. Finally the confidence prediction represents the IOU between the predicted box and any ground truth box.


#### Training

The initial CNN layers extract features from the image while the fully connected layers predict the output probabilities and coordinates. This network architecture is inspired by GoogleNet model for image classsfication adding the 1x1 reduction layers folllowed by 3x3 convolutional layers. The final output of network is the 7x7x30 tensor of predictions. This final layer predicts both class probabilities and bounding box coordinates. They normalize the bounding box width and height by the image width and height so that they fall between 0 and 1. They use linear activation function for the final layer and all other layers use the following learky rectified linear activation.

![architecture](https://user-images.githubusercontent.com/90513931/223003993-525694ff-410b-43b2-b61d-ad3394e5d8c4.png)


![loss function](https://user-images.githubusercontent.com/90513931/223004041-1e17ee77-d1f0-4e8c-9ca9-aa02c37e2b58.png)


They optimize for sum squared error in the output of their model. They use sum-squared error because it is easy to optimize, however it does not perfectly align with their goal of maximizing average precision. It weights localization error equally with classification error which may not be ideal. Also, in every image many grid celss do not contain any object. This pushes the "confidence" scores of those celss towards zero, often overpowering the gradient from cells that do conatin objects. This can lead to model instability, causing training to diverge early on. To remedy this, they increase the loss from bounding box coordinate predictions and decrease the loss from confidence predictions for boxes that don't contain objects. They use two parameters, lambda_coordinate and lambda_noobj to accomplish this. Sum squared error also equally weights errors in large boxes and small boxes.. This error metric should reflect that small deviations in large boxes matter less than in small boxes. To partially address this they predict the squear root of the bounding box width and height instead of the width and height directly. Note that the loss function only penalizes classification error if an object is present in that grid cell. It also only penalizes bounding box coordinate error if that predictior is "responsible" for the ground truth box.

![result](https://user-images.githubusercontent.com/90513931/223004043-e91ed2f2-8bea-4e86-a74c-a39a1a761273.png)


![qualitative results](https://user-images.githubusercontent.com/90513931/223004000-122438c8-fad7-484c-9233-eebb928d6033.png)
