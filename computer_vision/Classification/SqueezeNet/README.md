## SqueezeNet(AlexNet-Level accuracy with 50x fewer parameters and 0.5MB model size)

In paper, authors they focused on maintaining competitive accuracy, building CNN architecture with fewer parameters. They said that this fewer parameters model has several advantages.

- More efficient distributed training

- Less overhead when exporting new models to clients

- Feasible FPGA and embedded deploymend. 

To achieve this objectives, they take an existing CNN model and compress it in a lossy fashion. They use term _CNN microarchitecture_ to refer the particular organization and dimensions of the individual modules.

They introduce _Fire module_, new building block out of which to build CNN architecture. And they used this design strategies to construct _SqueezeNet_ whuch us comprised of Fire modules. They employ three main strategies when designing CNN architectures.

**1. Replace 3x3 filters with 1x1 filters**

They chose to make the majority of 1x1 filters, since a 1x1 filter has 9x fewer parameters.

**2. Decrease the number of input channels to 3x3 filters**

The total quantity of parameters is (number of input channels) x (number of filters) x (3 x 3). So they decreased thety number of input channels to 3x3 filters using squeeze layers.

**3. Downsample late in the network so that convolution layers have large activation**

K.He and H.Sun applied delayed down-sampling to four different CNN architectures, and these delayed downsampling led to higher classification accuracy.


### Fire module ###




