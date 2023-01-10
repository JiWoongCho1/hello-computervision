## AlexNet

In this paper authors proposed the various techniques for constraint the overfitting and down the error rate. They used the ImageNet dataset that is a dataset of over 15 million labeled high resolution images belonging to roughly 22,000 categories.

In architecture, it contains eight layers -- 5 convolutional, 3 fully connected.

ReLU --Tranditionally people used the 'tanh' activation function to train the model. But it was much slower than the non-santurating non-linearlity. 
ReLU was faster than their equivalents with tanh units.(6 times faster when training CIFAR-10 dataset)

GPU --They used the two GPUs. The parallelization scheme that they employ essentially puts half of the kernels on each GPU, with one additional trick: the GPUs communicate only in certain layers. 

Local Response Normalization --ReLUs have the desirable property that they do not require input normalization to prevent them from santurating.But they did implement the normalization. This sort of response normalization implements a form of lateral inhibition inspired by the type found in real neurons, creating compeetition for big activities amongst neuron outpus computed using different kernels. 

Overlapping Pooling --
