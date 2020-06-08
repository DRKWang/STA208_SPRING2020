Implametation Neural Network with Support Vector Machines (SVMs) for Classification
===


*This project was inspired by Y. Tang's [Deep Learning using Linear Support Vector Machines](https://arxiv.org/abs/1306.0239)
(2013), and by Abien Fred M. Agarap's [An Architecture Combining Convolutional Neural Network (CNN) and Support Vector Machine (SVM) for Image Classification](https://arxiv.org/pdf/1712.03541)


## Usage

First, clone the project.

Then, go to the repository's directory, and open the following notebooks to check the verification results on 3 different datasets ([MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR_10](https://www.cs.toronto.edu/~kriz/cifar.html), [fer_13]()).

- DNN on MNIST.ipynb
- CNN on CIFAR_10.ipynb
- 
## Implementations
To implement the algorithm that replacing the soft-max layer with a linear support vector machine, we took several steps as following:
#### 1 
We tried two different methods to construct DNN and CNN model. 

One of them is to use keras api to create a keras model. It is a stable and efficient way to implement neural network. However, there is no SVM model for DNN or CNN in keras and it is impossible for us to replace the softmax layer with the SVM layer. Even we try to define the SVM by ourselves to match the requests, we still fail to obtain the loss and the expression of weights and biases. So, we try the second method[1]: Constructing the NN model by defining each layer, loss function, and optimizer process. 

In normal CNN model, the last layer is the softmax function and the output is the probability of each class:
$$p_i = \frac{exp(a_i)}{\sum_{j}exp(a_j)}$$
and the predicted class $\hat{i}$
$$\hat{i} = \text{arg} \max p_i$$
We use cross-entropy loss here.

With Support Vector Machines, we delete the softmax layer and output the result from the last layer(Attention: No biases). Then, we define the soft margin loss[2]:
$$\min_w w^Tw + C\sum_{n=1}^{N}\max(1-w^T x_n y_n,0)^2$$
Actually, the primal form is L1_SVM with the standard hinge loss. But it is not differetiable so we use L2-SVM instead. 

To predict the class of data:
$$\hat{i} = \text{arg}_y \max (w^Tx)y$$
Here we only use linear SVM. 

#### 2
To implement Multiclass SVM, we use one-vs-rest approach. For K class problems, K linear SVMs are trained independently. The output of the $k$-th SVM is
$$a_k(x) = w^Tx$$
and the predicted class is
$$\text{arg}_k \max a_k(x)$$

#### 3
We also meet the problem that the graph is colorful. In this case, we consider one more parameter in our model, channel. If the data has k color channels,then we need k times parameters at the beginning. Actually, there are no much difference between the black-white data and colorful data. In implement this model for CIFAR-10 dataset.

## Results

#### 1. DNN with SVM v.s. DNN with softmax on MNIST

The hyperparameters used on MNIST were manually assigned, and not through optimization.

|Hyperparameters|CNN-Softmax|CNN-SVM|
|---------------|-----------|-------|
|Batch size|200|200|
|Learning rate|1e-3|1e-3|
|Steps|120000|120000|
|SVM C|N/A|2|


The experiments were conducted on a laptop computer with Intel Core(TM) i5-6300HQ CPU @ 2.30GHz x 4, 16GB of DDR3 RAM,
and NVIDIA GeForce GTX 960M 4GB DDR5 GPU.

![](figures/softmax1.png), ![](figures/SVM1.png), ![](figures/softmax2.png), ![](figures/SVM2.png)


**Figure 1. Training accuracy and loss of CNN-Softmax and CNN-SVM on [MNIST](http://yann.lecun.com/exdb/mnist/).**

#### 2. CNN with SVM v.s. CNN with softmax on CIFAR_10

**Figure 2. Training accuracy and loss of CNN-Softmax and CNN-SVM on [CIFAR_10](https://www.cs.toronto.edu/~kriz/cifar.html).**

#### 3. CNN with SVM v.s. CNN with softmax on CIFAR_10

**Figure 3. Training accuracy and loss of CNN-Softmax and CNN-SVM on [MNIST](http://yann.lecun.com/exdb/mnist/).**

![](figures/accuracy-loss-fashion.png)

**Figure 2. Training accuracy (left) and loss (right) of CNN-Softmax and CNN-SVM on image classification using [Fashion-MNIST](http://github.com/zalandoresearch/fashion-mnist).**

The red plot refers to the training accuracy and loss of CNN-Softmax, with a test accuracy of 91.86000227928162%.
On the other hand, the light blue plot refers to the training accuracy and loss of CNN-SVM, with a test accuracy of
90.71999788284302%. The result on CNN-Softmax corroborates the finding by [zalandoresearch](https://github.com/zalandoresearch) on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist#benchmark).


