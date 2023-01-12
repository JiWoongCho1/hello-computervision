## VGGNET

VGG Network was opened at 2015. It differs from other tranditional networks like AlexNet or Lenet, this network is more deep than others. Tranditionally, they used the large receptive field like 11 x 11, 5x5. But In this paper, authors argued that small receptive field and deep network is the better because of the computation, non linearity.

I implement the vgg16, vgg19 architecture using parameters that paper used.

Image size is 224 x 224 RGB image. They just did preprocessing that subtracting the mean RGB. They also utilized the 1 x 1 convolutioanl filters. Channels are started from 64 in the first layer and then increasing by a factor of 2 after each max-pooling layer until it reaches 512.
They incorporate three non-linear recification layers instead of a single one which makes the decision function more discriminative. --> increase the non linearity of the decision function.

In this paper, they used the ImageNet dataset which includes images of 1000 classes. 

Authors noted that using local response normalization(used in AlexNet) does not improve on the model.  And they confirmed thatdeep net with small filters outperforms a shallow net with larget filters(ex.AlexNet, LeNet). 
