## GoogleNet

GoogleNet is the one architecture using 'Inception module'. This 'Inception module' is so famous that this architecture also be called 'inceptionNet'. This naming is stemmed from that movie name 'Inception'. In this movie, one actor said that 'we need to go deeper', authors referred to this line.

When we ask a question that 'Deep network is the always best, yields good performance?' Answer is not because of the computational cost overfitting. For example, if two convolutional layers are chained, any uniform increase in the number of their filters results in a quadratic increase of computation.

To solving this problems, in paper, they suggest that both issues would be solved by moving from fully connected to sparsely connected architectures, even inside the convolutions.


In architecture, the main idea is how an optimal local sparse structure in a convolutional vision network can be approximated and covered by readily available dense components. They assumed that each unit from the earlier layer corresponds to some region of the input image and these units are grouped into filter banks. In the lower layers correlated units would concentrate in local regions. And in order to avoid path alignment issues, current incarnations of the Inception architecture are restricted to filter sizes 1 x 1, 3 x 3, 5 x 5, but they said that this is not neccessary, just convenience.
