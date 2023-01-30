## SENet

A central theme of computer vision research is the search for more powerful representations that capture only those properties of an image that are most salient for a given task, enabling imporoved performace. Recent research has shown that the representations produced by CNNs can be strengthened by integrating learning mechanisms into the network that help capture spatial correlations between features. In this paper, they investigated a different aspect of network desgin - relationship between channels. They introduced a new architectural unit, which they term the _Squeeze-and-Excitation_(SE) block, with the goal of improving the quality of representations produced by a network by explicitly modelling the interdependencies between the channels of its convolutional features.

![SEBlock](https://user-images.githubusercontent.com/90513931/215432224-aa9e396c-2ef6-41f1-bc06-fcb75bcacca6.png)

The design and development of new CNN architectures is a difficult engineering task, typically requiring the selection of many new hyperparameters and layer configurations. By contrast, the structure of the SE block is simple and can be used directly in existing state-of-the -art architectures by replacing components with their SE counterparts, where the performance can be effectively enhanced. SE blocks are also computationally lightweight and impose only a slight increase in model complexity and computational burden. Consequently their best model ensemble achieves a 2.251% top-5 error on the test set. This represents roughly a 25% relative improvement.

#### Squeeze: Global information embedding

Each of the learned filters operates with a local receptive field and consequently each unit of the transformation output U is unable to exploit contextual information outside of the region. To mitigate this problem, they proposed to _squeeze_ global spatial information into a channel descriptor. This is achievend by using global average pooling to generate channel-wise statistics. They opt for the simplest aggregation technique, global average pooling, nothing that more sophisticated strategies could be employed here as well. And they examine the significance of using global average pooling as opposed to global max pooling as their choice of squeeze operator. While both max and average pooling are effective, average pooling achieves sligtly better performance, justifying its selection as the basis of the squeeze operation.

#### Excitation: Adaptive Recalibration

To make use of the information aggregated in the _squeeze_ operation, they follow it with a second operation which aims to fully capture channel-wise dependencies. To fulfill this objective, the function must meet two criteria: first, it must be flexible. Second, it must learn a non-mutually exclusive relationship since we would like to ensure that multiple channels are allowed to be emphasised. To meet these criteria, they opt to employ a simple gating mechanism with a sigmoid activation. And they observed that exchanging the sigmoid for tanh slightly worsens performance, while using ReLU is dramatically worse and in fact causes the performance of SE-ResNet-50 to drop below that of the ResNet-50 baseline.

![tanh_maxpooling](https://user-images.githubusercontent.com/90513931/215432225-11f8b386-31bc-44d0-86c2-7070d7136403.png)

When using SE blocks in earlier layers, it excites informative features in a class agnostic manner, strengthening the shared low level representations. In later layers, the SE blocks become increasingly specialized, and respond to different inputs in a highly class-specific manner. Also they observed that SE blocks bring improvements in performance on the non-residual settings.

The SE block can be integrated into standard architectures such as VGGNet by insertion after the non-linearity following each convolution. Moreover, the flexibility of the SE block means that it can be directly applied to tranformations beyond standard convolutions. And it must offer a good trade-ff between imporved performance and increased model complexity corresponding to a 0.26% relative increase over the original ResNet-50. And they consider two representative architectures from the class of mobile-optimized networks, MobileNet, ShuffleNet and they show that SE blocks consistently improve the accuracy by a large margin at a minimal increase in computational cost.

![application of SEBlock](https://user-images.githubusercontent.com/90513931/215432217-fc4f0072-9d31-47b3-b68d-7e1585129e50.png)

![mobile application](https://user-images.githubusercontent.com/90513931/215432223-4f3ba17f-428f-4eae-8afe-50ee6226913a.png)

They further assess the generalization of SE blocks on the task of object detection using the COCO dataset. They use the Faster R-CNN detectiob framework as the basis for evaluating their models. And it shows a relative 6.3% improvement on COCO's standard AP metric and by 3.1% on AP@Iou = 0.5

They performed an ablation study to assess the influence of the location of the SE block when integrating it into existing archtectures. They observed that the SE-PRE, SE-Identity and proposed SE block each perform similarly well, while usage of the SE-POST block leads to a drop in performance. This experiment suggets that the performance improvements produced by SE units are fairly robust to their location.

![error rate](https://user-images.githubusercontent.com/90513931/215432222-b7c5e5d2-1400-40b3-96fe-153c25663939.png)





