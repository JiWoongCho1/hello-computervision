## FCN


Convnets are not only improving for whole-image classification, but also making progress on local tasks with structured output. These include advances in bounding box object detection, part and key point prediction and local correspondence. The natural next step in the progression from coarse to fine inference is to make a prediction at every pixel.

They show that a fully convolutional network(FCN) trained end-to-end, pixels-to-piels on semantic segmentation exceeds the state-of-the-art without other machinery. To implement the task, they use the upsampling layers that enable pixelwise prediction and skip architectures that combine deep, coarse, semantic information and shallow, fine, appearance information. 

Typical AlexNet, VGGNet take fixed sized inputs and produce nonspatial outputs(output layer is fully connected layer). The fully connected layers of these nets have fixed dimensions and throw away coordinates. However, these fully connected layers can also be viewed as convolutions with kernels that cover their entire input regions. Doing so casts them into fully convolutional networks that take input of any size and output classification maps. Furthermore, while the resulting maps are equivalent to the evaluation of the original net on particular input patches, the computation is highly amortized over the overlapping regions of those patches which 5 times faster than the naive approach. And the spatial output maps of these convolutionalized models make them a natural choice for dense problems like semantic segmentation.

(figure2 사진)

One way to connect coarse outputs to dense pixels is interpolation. For instance, simple bilinear interpolation icomputes each output y{ij} from the nearest four inputs by a linear map that depends only on the relative positions of the input and output cells. In a sense, upsampling with factor 'f' is convolution with a fractional input stride of 1/f. So long as f is integral, a natrual way to upsample is therefore backwards convolution(called deconvolution) with an output stride of f. Such an operation is trivial to implement, since it simply reverses the forward and backward passes of convolution. Note that the deconvolution filter in suvh a layer need not be fixed, but can be learned. In their experiments, they find that in-network upsampling is fast and effective for learning dennse prediction.

(figure 3 사진)
