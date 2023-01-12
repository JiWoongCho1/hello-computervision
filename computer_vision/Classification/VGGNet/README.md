## VGGNET

VGG Network was opened at 2015. It differs from other tranditional networks like AlexNet or Lenet, this network is more deep than others. Tranditionally, they used the large receptive field like 11 x 11, 5x5. But In this paper, authors argued that small receptive field and deep network is the better because of the computation, non linearity.

![architecture](https://user-images.githubusercontent.com/90513931/212109755-7240bfa6-52ff-4c48-b610-c54b8e545508.png)

Image show that VGGNet can have various layers that decide the name of the architecture. 'D' is VGG16, 'E' is VGG19 which are mostly used. 

I implement the vgg16, vgg19 architecture using parameters that paper used.


Image size is 224 x 224 RGB image. They just did preprocessing that subtracting the mean RGB. They also utilized the 1 x 1 convolutioanl filters. Channels are started from 64 in the first layer and then increasing by a factor of 2 after each max-pooling layer until it reaches 512.
They incorporate three non-linear recification layers instead of a single one which makes the decision function more discriminative. --> increase the non linearity of the decision function.

In this paper, they used the ImageNet dataset which includes images of 1000 classes. 

Authors noted that using local response normalization(used in AlexNet) does not improve on the model.  And they confirmed thatdeep net with small filters outperforms a shallow net with larget filters(ex.AlexNet, LeNet).

![acc_rate](https://user-images.githubusercontent.com/90513931/212109771-858ad53a-59cc-46fb-bcd4-9cb7afa7352a.png)

![comparison](https://user-images.githubusercontent.com/90513931/212109784-c168ca25-84d0-4b5f-904d-33f2c7342ecb.png)


This image show that VGG is better than other architecture.
