## ResNet(Deep Residual Learning for Image Recognition)

Tranditionally simple deeper network has higher training errpor and rest error than shallow network. This is caused by degradation problem that accuracy gets santurated and degrade rapidy. These problems indicate that not all systems are similarly easy to optimize. 

![comparison_deep-shallow](https://user-images.githubusercontent.com/90513931/212635880-285d2639-14e7-4600-a576-492d22b4b54f.png)

So they addressed the degradation problem by introducing a 'deep residual learning' framework. They assumed that instead of each few stacked layers directly, let layers fit a residual mapping. In this paper, denoting the desired underlaying mapping as H(x), let the stacked nonlinear layers fit another mapping of F(x) = H(x) -x
Thiis formulation can be realized by feedforward neural networks with 'shortcut connections' This shortcut connection simply perform 'identity mapping'(conceptually same but effect is different). 

![skip-connection](https://user-images.githubusercontent.com/90513931/212635917-007914c3-fad3-4d1d-b6c5-45d6215ef550.png)

When they implement this connections, models are easy to optimize, enjoy accuracy gains from greatly increased depth. The degradation problem suggests that the solvers might have difficulties in approximating identity mappings by multiple nonlinear layers. With the residual learning reformulation, but, the solvers may simply drive the weights of the multiple nonlinear layers toward zero to approach identity mappings.

In network architectures, they noticed that their models have fewer filters and lower complexity than VGGNets. Alos their models are trained with BN, which eacures forward propagated signals to have non-zero variance. In comparison, they note that the 18-layer plain/residual nets are comparably accuracte but the 18-layer ResNet converges faster. And they also stacked 152 layers, but the 152 layer ResNet still has lower complexity than VGGNets and more accuracte than the 34 layer ones by considerable margins.

![comparison-18-34](https://user-images.githubusercontent.com/90513931/212635895-99c718d6-696d-45ac-909a-5dd21dc878a1.png)
