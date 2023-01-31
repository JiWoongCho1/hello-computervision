## DenseNet

As CNNs become increasingly deep, a new research problem emerges. Weights and informations are pass through many layers and it can vanish and 'wash out' by the time it reaches the end of the network. So in this paper, they proposed a architecture to ensure maximum information flow between layers in the network, they connected all layers. This method is similar to ResNet, but they never combine features through summation before they are passed into a layer, they combine features by concatenation them, they refered to this approach as _Dense Convolutional Network_ 

<img width="339" alt="architecture" src="https://user-images.githubusercontent.com/90513931/213763955-b7ad5409-e868-44d0-a165-5c421d10be19.png">

This architecture requires fewer parameters than tranditional convolutioanl(in particular, ResNet). Each layer read the state from its preceding layer and writres to the subsequent layer. This _feature reuse_ yield condensed models that are easy to train and highly parameter efficiently. Concatenation feature-maps learned by different layers increases variation in the input of subsequent layers and imporves efficiency, which are similar to InceptionNet but DenseNets are simpler and more efficient. One explanation for concatenation is that each layer has access to all the preceding feature-maps in its block and to the network's "collective knowledge".

In contrast, ResNet did the identity function and the output are combined by summation, which may impede the information flow in the network.

To faclitate down-sampling in their architecture, they divide the network into multiple densely connected _dense blocks_. They refer to layers between blocks as _transition layers_, which do convolution and pooling.

To further improve model compactness, they reduced the number of feature-maps at transition layers. (In paper, denoting theta, 1 is unchanged)

They show that DenseNet has fewer parameters and efficient. This fewer parameters yield that less prone to overfitting. They achieve same level of accuracy and only requires around 1/3 of the parameters of ResNets.

![plot](https://user-images.githubusercontent.com/90513931/213763962-08720a91-5b16-4d0e-91d4-2cef991e6b85.png)

