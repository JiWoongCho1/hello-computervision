## DenseNet

As CNNs become increasingly deep, a new research problem emerges. Weights and informations are pass through many layers and it can vanish and 'wash out' tby the time it reaches the end of the network. So in this paper, they proposed a architecture to ensure maximum information flow between layers in the network, they connected all layers. This method is similart to ResNet, but they never combine features through summation before they are passed into a layer, they combine features by concatenation them, they refered to this approach as _Dense Convolutional Network_ 

This architecture requires fewer parametersthan tranditional convolutioanl(in particular, ResNet). Each layer read the state from its preceding layer and writres to the subsequent layer. This _feature reuse_ yield condensed models that are easy to train and highly parameter efficiently. Concatenation feature-maps learned by differend layers increases variation in the input of subsequent layers and imporves efficiency, which are similar to InceptionNet but EnseNets are simpler and more efficient. One explanation for concatenation is that each layer has access to all the preceding feature-maps in its block and to the network's "collective knowledge".

In contrast, ResNet did the identity function and the output are combined by summation, which may impede the information flow in the network.

To faclitate down-sampling in theirt architecture, they divide the network into multiple densely connected _dense blocks_. They refer to layers between blocks as _transition layers_, which do convolution and pooling.

To further improve model compactness, they reduced the number of feature-maps at transition layers. (In paper, denoting theta, 1 is unchanged)

They show that DenseNet has fewer parameters and efficient. This fewer parameters yield that less prone to overfitting. They achieve same level of accuracy and only requires around 1/3 of the parameters of ResNets.

