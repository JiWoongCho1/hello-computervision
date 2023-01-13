## GoogleNet

GoogleNet is the one architecture using 'Inception module'. This 'Inception module' is so famous that this architecture also be called 'inceptionNet'. This naming is stemmed from that movie name 'Inception'. In this movie, one actor said that 'we need to go deeper', authors referred to this line.

When we ask a question that 'Deep network is the always best, yields good performance?' Answer is not because of the computational cost overfitting. For example, if two convolutional layers are chained, any uniform increase in the number of their filters results in a quadratic increase of computation.

To solving this problems, in paper, they suggest that both issues would be solved by moving from fully connected to sparsely connected architectures, even inside the convolutions.


In architecture, the main idea is how an optimal local sparse structure in a convolutional vision network can be approximated and covered by readily available dense components. They assumed that each unit from the earlier layer corresponds to some region of the input image and these units are grouped into filter banks. In the lower layers correlated units would concentrate in local regions. And in order to avoid path alignment issues, current incarnations of the Inception architecture are restricted to filter sizes 1x1, 3x3, 5x5, but they said that this is not neccessary, just convenience. Additionally, since pooling operations have been essential for te success in current state of the art convolutional networks, it suggests that addng an alternative parallel pooling path in each such stage.

One of the main beneficial aspects of this architecture is that it allows for increasing the number of units at each stage significantly without an uncontrolled blow-up in computational complexity. Another practically useful aspect of this design is that it aligns with the intuition that visual information should be processed at various scales and then aggregated so that the next stage can abstract features from different scales simultaneously. 

They use rectified linear activation function and '3x3 reduce', '5x5 reduce' stands for the number of 1x1 filters in the reduction layer used befor the 3x3 and 5x5 convolutions. The network is 22 layers deep when counting only layers. Although depth could have concern that it can be not propage the gradients back through all the layers. But in this architecture they noted that layers in the middle of the network could produce features that is very discriminative. So they added the auxiliary classifiers connected to these intermediate layers. During training, their loss added to the total loss of the network and be discarded when training is starting.

