## Lenet 5

In this paper, they have the messages that better pattern recognition systems that can be built by relying more on automatic learning.

![Lenet](https://user-images.githubusercontent.com/90513931/211243841-0cc19e17-6e46-442b-9b50-d356b62312d7.png)

In my code, do my best to implement to original structure of the Lenet-5. But unfortunately i couldn't one structure that S2-C3 details.(In this layers, they are not fully connected, partially connected)

And i visualize the kernel of each layers and feature maps that pass through the features. I think this is helpful for understaning the convolutional layers. 

They adapted the gradient based learning for minimizing the loss much easily and small variations.

Also they trained with multi layer networks with complexity, high dimensional. In this structure it combines three architectural ideas to ensure some degree of shift scale, and distortion invariance: local receptive fields, shared weights, sub-sampling.

The idea of connecting unit units to local receptive fields on the input goes back to the Perceptron in the early 60s, and was almost simultaneously with Hubel and Wiesel's discovery of locally-sensitive, orientation-selective neurons in the cat's visual system. With local receptive fields, neurons can extract elementary visual features such as oriented edges, end-points, corners. These features are then combined by the subsequent layers in order to detect higher-order features.

The kernel of the of the convolution is the set of connection weights used by the units in the feature map. An interestin gproperty of the convolutional layers is that if the input image is shifted, the feature map output will be shifted by the same amount, but will be left unchanged otherwise. This propery is at the basis of the robustness of convolutional networks to shifts and distortions of the input.

A simple way to reduce the precision with which the position of distinctive features are encoded in a feature map is to reduce the spatial resolution of the feature map.
