## AlexNet

In this paper authors proposed the various techniques for constraint the overfitting and down the error rate. They used the ImageNet dataset that is a dataset of over 15 million labeled high resolution images belonging to roughly 22,000 categories. But i used the Fashion-MNIST dataset because of the time, GPU.  

In architecture, it contains eight layers -- 5 convolutional, 3 fully connected.


![Alexnet](https://user-images.githubusercontent.com/90513931/211596630-1b4df560-0000-4e5a-8ad6-cc95a6dbdb5d.png)


#### _ReLU_
Tranditionally people used the 'tanh' activation function to train the model. But it was much slower than the non-santurating non-linearlity. 
ReLU was faster than their equivalents with tanh units.(6 times faster when training CIFAR-10 dataset)

#### _GPU_
They used the two GPUs. The parallelization scheme that they employ essentially puts half of the kernels on each GPU, with one additional trick: the GPUs communicate only in certain layers. (Image is that visualizing the each GPU's feature maps. In this image, we could identify that each GPU train the other features.)

![filters](https://user-images.githubusercontent.com/90513931/211596400-54538922-1952-49b9-8074-69e7667d753b.png)


#### _Local Response Normalization_
ReLUs have the desirable property that they do not require input normalization to prevent them from santurating.But they did implement the normalization. This sort of response normalization implements a form of lateral inhibition inspired by the type found in real neurons, creating compeetition for big activities amongst neuron outpus computed using different kernels. 

![LocalResponseNorm](https://user-images.githubusercontent.com/90513931/211596424-68a592af-6904-457f-843c-17600fd20ae2.png)


#### _Overlapping Pooling_ 
Tranditionally, the neighborhoods summarized by adjacent pooling units do not overlap. To be more precise, a pooling layer can be thought of as consisting of a grid of pooling units spaced 's' pixels apart, each summarizing a neighborhood of size 'z x z' centered at the location of the pooling unit. They set s < z(s = 2, z = 3)/ This scheme reduce the error rate. They generally observed during training that models with overlapping pooling find it slightly more difficult to overfit.

#### _Overall Architecture_
the output of the last fully connected layer is fed to a 1000-way softmax which produces a distribution over the 1000 class labels. its network maximizes the multinomial logistic regression objective, which is equivalent to maximizing the average across training cases of the los-probability of the correct label under the prediction distribution. In original paper, first convolutional layer filters the 224 x 224 x 3 input image with 96 kernels of size 11 x 11 x 3 with stride 4. Second convolutional layer takes as input the output of the first convolutional layer and filters it with 256 kernels of size 5 x 5x 28. Remainder of the kernel size, feature map can be identified at my code.

#### _Data Augmentation_ 
One way of the avoid the overfitting is enlarge the dataset. So they did this by extracting random 224 x 224 patches from the 256 x 256 images and training their network on these extracted patches. And they generated image translations and horizontal reflections. Also they generated method that alter the intensities of the RGB channels in training images. The performed PCA on the set of RGB pixel values throughout the ImageNet training set.

#### _Dropout_ 
It consists of setting to zero the output of each hidden neuron with probability 0.5. The neurons which are "dropped out" in this way do not contribute to the forward pass and do not participate in backpropagation. They used dropout in the first two fully-connected layers. Without dropout, the network exhibits substational overfitting. Dropout roughly doubles the number of iterations required to converge.

