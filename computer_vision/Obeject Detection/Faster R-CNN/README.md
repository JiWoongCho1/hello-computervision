## Faster R-CNN

Although region-based CNNs were computationally expensive as originally developed, their cost has been drastically reduced thanks to sharing convolutions across proposals. The latest incarnation, Fast R-CNN achieves near real time rates using very deep networks, when ignoring the time spent on region proposals. Selective search, one of the most popular methods, greedily merges superpixels based on engineered low level features. Yet when compared to efficient detection networks, Selective search is an order of magnitude slower, at 2 seconds per image in a CPU implementation. EdgeBoxes currently provides the best tradeoff between proposal quality and speed, at 0.2 seconds per image. Nevertheless, the region proposal step still consumes as much running time as the detection network. 
 
 In this paper, they show that an algorithmic change leads to an elegant and effective solution where proposal computation is nearly cost-free given the detection network's computation. To this end, they introduce novel _Region Proposal Network_(RPNs) that share convolutional layers with state-of-the-art object detection networks. By sharing convolutions at test time, the marginal cost for computing proposals is small.(10ms per image). Their observation is that the convolutional feature maps used by region-based detectors, like Fast R-CNN, can also be used for generating region proposals. On top of these convolutional features, they construct an RPN by adding a few additional convolutional layers that simultanesously regress region bounds and objectness scores at each location on a regular grid. The RPN is thus a kind of fully convolutional network and can be trained end-to-end specifically for the task for generating detection proposals. 
 
 RPNs are designed to efficiently predict region proposals with a wide range of scales and aspect ratios. In contrast to prevalent methods that use pyramids of images or pyramids of filters, they introduce novel 'anchor' boxes that serve as references at multiple scales and aspect ratios. This model performs well when trained and tested using single scale images and thus benefits running speed. To unify RPNs with Fast R-CNN object detection networks, they propose a training scheme that alternates between fine-tuning for the region proposal task and then fine-tuning for object detection, while keeping the proposals fixed. RPNs completely learn to proposed regions from data, and thus can easily benefit from deeper and more expensive features.
 
 Their object detection system, called Faster R-CNN, is composed of two modules. The first module is a deep fully convolutional network that proposes regions, and the second module is the Fast R-CNN detector that uses the proposed regions. The entire systems is a single, unified network for object detection. Using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN module tells the Fast R-CNN module where to look.
 
 ![architecture](https://user-images.githubusercontent.com/90513931/222379227-ba9acbb4-b614-4e82-9168-e4d148f634a7.png)
 
 
 #### Region Proposal Networks
 
 A Region Proposal Network takes an image as input and output a set of rectangular object proposals, each with an objectness score. Because their ultimate goal is to share computation with a Fast R-CNN object detection network, they assume that both nets share a common set of convolutional layers(In this experiment they apply and use a VGG16). To generate region proposals, they slide a small network over the convolutional feature map output by the last shared convolutional layer. This small network takes as input an _n x n_ spatial window of the input convolutional feature map. In this paper they use n = 3. This mini network is showed in next figure and note that because the mini-network operates in a sliding-window fashion, the fully connected layers are shared across all spatial locations. 

![RPN and anchor](https://user-images.githubusercontent.com/90513931/222379223-030445b8-5a26-41b1-9bba-7e061ce617bf.png)

 
 At each sliding-window location, they simultaneously predict multiple region proposals. So the reg layer has 4k outputs encoding the coordinates of k boxes, and the cls layer outputs 2k scores that estimate probability of object or not object for each proposal. The k proposals are parameterized relative to k reference boxes, which they call _anchors_. An anchor is centered at the sliding window in question, and is associated with a scale and aspect ratio.
 
 An important property of this approach is that it is _translation invariant_. If one transflates an object in an image, the proposal should translate and the same function should be abke to predict the proposal in either location. As a comparison, the Multibox method uses kmeans to generate 800 anchors, which are not translation invariant. The translation invariant property also reduces the model size. So authors expect their method to have less risk of overfitting on small datastes.
 
 For training RPNs, they assign a binary class label to each anchor, positive/negative using IoU.

![loss function](https://user-images.githubusercontent.com/90513931/222379219-7dd885f7-e51a-4275-929e-2948ef37d446.png)
 It is possible to optimize for the loss functions of all anchors, but this will bias towards negative samples as they are domninate. Instead, they randomly sample 256 anchors in an image to compute the loss function of a mini-batch, where the sampled positive and negative anchors have a ratio of up to 1:1.
 
 #### Sharing features for RPN and Fast R-CNN
 
 To achieve this, they adopt the 4-step alternate training. First step, they train the RPN as above described. This network is initialized with an ImageNet pre-trainedd model and fine-tuned end-to-end for the region proposal task. Second step, they train a seperate detection network by Fast R-CNN using the proposals generated by the step-1 RPN. At this point the two networks do not share convolutional layers. Third step, they use the detector network to initialize RPN training, but fix the shared convolutional layers and only fine-tune the layers unique to RPN. Now the two networks share convolutional layers. Finally, keeping the shared convolutional layers fixed, they fine-tune the unique layers of Fast R-CNN.
