## DeepLab v1


Over the past two years DCNNs have pushed the performance of computer vision systems to soaring heights on a broad array of high-level problems, including image classification, object detection, fine-grained categorization. This success can be partially attributed to the built-in invariance of DCNNs to local image transformations, which underpins their ability to learn hierarchical abstractions of data. While this invariance is clearly desirable for high-level vision tasks. It can hamper low-level tasks, such as pose estimation, semantic segmentation where they want precise localization, rather than abstraction of spatial details.

So they employ atrous(dilated) algorithm originally developed for efficiently computing the undecimated discrete wavelet transform. This allows efficient dense computation of DCNN responses in a scheme substantially simpler than earlier solutions to this problem. Another problem relates to the fact that obtaining object-centric decisions from a classifier requires invariance to spatial transformations, inherently limiting the spatial accuracy of the DCNN model. They boost their model's ability to capture fine details by employing a fully-connected Conditional Random field(CRF). These main advantages of their "DeepLab" systems are speed, accuracy, simplicity.



They skip subsampling after the last two max-poolig layers in the network and modify the convolutional filters in the layers that follow them by introducing zeros to increase their length. They can implement this more efficiently by keeping the filters intact and instead sparsely sample the feature maps on which they are applied on using an input stride of 2 or 4 pixels, respectively. This approach is generally applicable and allow them to efficiently compute dense CNN feature maps at any target subsampling rate without introducing any approximations.

![atrous algorithm](https://user-images.githubusercontent.com/90513931/223952260-3740152a-f0d5-4a30-aa97-09e447c64e07.png)


Another key ingradient in re-purposing their network for dense score computation is explicitly controlling the network's receptive field size. They implement spatially subsampling the first FC layer to 4x4 spatial size. This has reduces the receptive field of the network down to 128x128 or 308x308 and has reduced computation time for the firer FC layer by 2-3 times.


As illustrated in next figure, DCNN score maps can reliably predict the presence and rough position of objects in an image but are less well suited for pin-pointing their exact outline. There is natural trade-off between classification accuracy and localization accuracy with convolutional networks: Deeper models wth multiple max-pooling layers have proven most successful in classification tasks, however their increased invariance and large receptive fields make the problem of inferring position fromthe scores at their top output levels more challenging. So they pursue a novel alternative direction based on coupling the recognition capacity of DCNNs and the fine-grained localization accuracy of fully connected CRFs and show that it is remarkably successful in addressing the localization challenge, producing accurate semantic segmentation results and recovering object boundaries at a level of detail.

![score map](https://user-images.githubusercontent.com/90513931/223952254-28c16f0e-5912-4510-b410-cb0ba3801cbb.png)


Tranditionally, conditional random fields(CRFs) have been employed to smooth noisy segmentation maps. In this regime, using short-range CRFs can be deterimental, as out goal should be to recover detailed local structure rather than smooth it. In conjuction to local-range CRFs can potentialy improve localization but still miss thin-structures and typically requires solving an expensive discrete optimization problem. To overcome these limitations of short-range of CRFs, they integrate into their system the fully connected CRF model. They use unary term and pairwias term and pairwise term is used for each pair of pixels _i, j_ in the image no matter how far from each other they lie, the model's factor graph is fully connected.

![energy function](https://user-images.githubusercontent.com/90513931/223952264-97c4854d-ff39-4d12-a572-bc9c32d6c18f.png)

![pairwise term](https://user-images.githubusercontent.com/90513931/223952266-96d9eda6-565d-4af2-90ad-50610319d36d.png)


They have also explored a multi scale prediction method to increase boundary localization accuracy. Specifically, they attach to the input image and the output of each of the first four max pooing layers a two layer MLP whose feature map is concatenated to the main network's last layer feature map. The aggregate feature map fed into the softmax. Introducing these extra direct connections from fine-resolution layers improves localization performance, yet the effect is not as dramatic as the one obtained with the fully connected CRF.

![comparison](https://user-images.githubusercontent.com/90513931/223952261-750046a1-2396-4559-bf0c-550af909d9b2.png)

![performance1](https://user-images.githubusercontent.com/90513931/223952267-aadd01ff-ddc5-4745-a16e-f468f91584b7.png)


