## DenseNet

As CNNs become increasingly deep, a new research problem emerges. Weights and informations are pass through many layers and it can vanish and 'wash out' tby the time it reaches the end of the network. So in this paper, they proposed a architecture to ensure maximum information flow between layers in the network, they connected all layers. This method is similart to ResNet, but they never combine features through summation before they are passed into a layer, they combine features by concatenation them, they refered to this approach as _Dense Convolutional Network_ 
