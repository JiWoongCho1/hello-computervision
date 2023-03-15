## SPPNet


There is a technical issue in the training and testing of the CNNs: the prevalent CNNs require a fixed input size, which limits both the aspect ratio and the scale of the input image. When applied to images of arbitrary sizes, current methods mostly fit the input image to the fixed size, either via cropping or via warping as shown in a figure. But the cropped region may not contain the entire object, while the warped content may result in unwanted geometric distortion. Recognition accuracy can be compromised due to the content loss or distortion. 


![problem](https://user-images.githubusercontent.com/90513931/220559935-8d8b9187-20fc-4686-be71-cb4ed64c5135.png)

So why do CNNs require a fixed input size? A CNN mainly consists of two parts: convolutional layers, and fully connected layers. In fact, convolutional layers do not require a fixed sizes, on the otherhand, the fully connected layers need to have fixed size/length input by their definition. In this paper, they introduce a _spatial pyramid pooling_ layers to remove the fixed size constraint of the network. Specifically, they add an SPP layer on top of the last convolutional layer. The SPP layer pools the features and generates fixed length outputs, which are then fed into the fully connected layers. In other words, they perform some information  "aggregation" at a deeper stage of the network hierarch to avoid the need for cropping or warping ar the beginning. Next figure hows the change of the network architecture by introducing the SPP layer. They call the new network structure SPPnet.

![network](https://user-images.githubusercontent.com/90513931/220559933-7552ed28-9402-4ef3-822f-dacfe975e116.png)


It partitions the image into divisions from finer to coarser levels, and aggregates local features in them. SPP has long been a key component in the leading and competition winning systems for classification and detection before the recent prevalence of CNNs. Nevertheless, SPP has not been considered in the context of CNNs. They note that SPP has several remarkable properties for deep CNNs: 1) SPP is able to generate a fixed length output regardless of the input size, while the sliding window pooling used in the previous deep networks cannot. 2) SPP uses multi level spatial bins, while the sliding window pooling uses only a single window size. Multi level pooling has been shown to be robust to object deformations. 3) SPP can pool features extracted at variables scales thanks to the flexibility of input scales. Through experiments they show that all these factors elevate the recognition accuracy of deep networks. SPP net not only makes it possible to generate representations from arbitrarily sized images/windows for testing, but also allows them to feed images with varying sizes or scales during training. Training with variable size images increases scale invariance and reduces overfitting. SPPnet also shows state-of-the-art classification results on Caltech101 and Pascal VOC2007 using only a single full image representation and no fine tuning. Also shows great strength in object detection. In the leading object detection method R-CNN, the feature computation in R-CNN is time consuming,  because it repeatedly applies the deep convolutional networks to the raw pixels of thousands of warped region per image. In this paper, they show that they can run the convolutional layers only once on the entire image and then extract features by SPPnet on the feature maps.


Spatial pyramid pooling can maintain spatial information by pooling in local spatial bins. These spatial bins have sizes proportional to the image size, so the number of bins is fixed regardless of the image size. To adopt the deep network for image of arbitrary size, they replace the last pooling layer with a spatial pyramid pooling layer. Next figure illustrates their method.


![SPP layer](https://user-images.githubusercontent.com/90513931/220559927-e23562fc-639c-4a1b-a99f-8bdb7f52417e.png)

With spatial pyramid pooling, the input image can be of any sizes. This not only allows arbitrary aspect ratios, but also allows arbitrary scales. Interestingly, the coarsest pyramid level has a single bin that covers the entire image. This is infact a "global pooling" operation, which is also investigated in several concurrent works. It is also used to reduce the model size and also reduce the overfitting.

![error rate](https://user-images.githubusercontent.com/90513931/220559932-a50728a4-1945-455e-a6ac-2f7d9f3192ff.png)



![result](https://user-images.githubusercontent.com/90513931/220559919-2b6cd081-54ed-4774-b556-01c239526681.png)
