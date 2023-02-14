## Swin Transformer


Beginning with AlexNet and its revolutionary performance on the ImageNet image classification challenge, CNN architectures have evolved to become increasingly powerful through greater scale, more extensive connections and more sophisticated forms of convolution. In this paper, they seek to expand the applicability of Transformer such that it can serve as a general purpose backbone for computer vision, as it does for NLP and as CNNs do in vision. Unlike the work tokens that serve as the basic elements of processing on language Transformers, visual elements can vary substantially in scale, a problem that receives attention in tasks such as object detection. In existing Transformer-based models, tokens are all of a fixed scale, a property unsuitable for these vision applications. Another difference is the much higher resolution of pixels in images compared to words in passages of text. There exist many vision tasks such as semantic segmentation that require dense prediction at the pixel level, and this would be intractable for Transformer on high resolution images, as the computational complexity of its self attention is quadratic to image size. To overcome these issues, they propose a general purpose Transformer backbone, called Swin Transformer, which constructs hierarchical feature maps and has linear computational complexity to image size. This figure show the hierarchical representation by starting from small sized patches and gradually merging neighboring patches in deeper Transformer layers. The number of patches in each window is fixed, and thus the complexity becomes linear to image size.

![hierarchical](https://user-images.githubusercontent.com/90513931/218647910-f9cb96ac-ea56-40b2-be28-24b8fc2f3ed3.png)



First, splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT. Each patch is treated as a 'token' and its feature is set as a concatenation. Several Transformer blocks with modified self-attention computation are applied on these patch tokens. The Transformerblocks maintain the number of tokens(H/4 x W/4), and together with the linear embedding are referred to as 'Stage 1'. To produce a hierarchical representation, the number of tokens is reduced by patch merging layers as the network gets deeper. The first patch merging layer concatenates the features of each group of 2x2 neighboring patches, and applies a linear layer on the 4C-dimensional concatenated features. This reduces the number of tokens by a multiple of 2x2 = 4(2x downsampling of resolution), and the output dimension is set to 2C. Swin Transformer blocks are applied afterwards for feature transformation, with the resolution kept at H/8 x W/8. This first block of patch merging and feature transfromation is denoted as 'Stage 2'. The procedure is repeated twice, as 'Stage 3' and 'Stage 4', with output resolutions of H/16 x W/16 and H/32x H/32. These stages jointly produce a hierarchcal representation. As a result, the proposed architecture can conveniently replace the backbone networks in existing methods for various vision tasks. Swin Transformer is built by replacing the standard multi-head self attention module in a Transformer block by a module based on shifted windows, with other layers kept the same

![architecture](https://user-images.githubusercontent.com/90513931/218647903-8d7d3784-daed-467c-a3c3-934e33d5546d.png)


#### Shifted Window based Self-Attention

The standard Transformer architecture and its adaptation for image classfication both conduct global self-attention, where the relationshups between a token and all other tokens are computed. The global computation leads to quadratic complexity with respect to the number of tokens, making it unsuitable for many vision problems requiring an immense set of tokens for dense prediction or to represent a high resolution image. The window-based self attention module lacks connections across windows, which limits its modeling power. To introduce cross-window connections while maintaining the efficient computation of non-overlapping windows, they propose a shifted window partitioning approach which alternates between two partitioning configutations in consecutive Swin Transformer blocks. The shifted window partitioning approach introduces connections between neighboring non-overlapping windows in the previous layer and is found to be effective in image classification, object detection and semantic segmentation.

![shifted window](https://user-images.githubusercontent.com/90513931/218647901-f50aea05-f03d-432c-9adb-446024345314.png)

#### Efficient batch computation for shifted configuration

When the number of windows in regular partiotioning is small, the increased computation with this naive solution is considerable. So they propose a more efficient batch computation approach by cyclic shifting toward the top left direction. After this shift, a batched window may be composed of several sub windows that are not adjacent in the feature map, so a masking mechanism is employed to limit self-attention computation to within each sub-window. With the cyclic shift, the number of batched windows remains the same as that of regular window partitioning, and thtus is also efficient.

![cycle shift](https://user-images.githubusercontent.com/90513931/218647908-a32593f3-0adc-490e-b55b-549b01092e5c.png)



This figure show the comparison to SwinTransformer and ResNet on the object detection framework. SwinTransformer gbrings consistent 3.4~4.2 box AP gains over ResNet, with slightly larger model size, FLOPs and latency. Also compares their best results with those of previous state of the art models. Their best model achieves 58.7 box AP and 51.1 mask AP on COCO test-dev, surpassing the previous best results by 2.7 box AP and 2.6 mask AP.

![result](https://user-images.githubusercontent.com/90513931/218647894-972b7ba5-b88d-476b-8b34-be88a9fbb2d1.png)
