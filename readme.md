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


## Results

#### 1. DNN with SVM v.s. DNN with softmax on MNIST

The hyperparameters used on MNIST were manually assigned, and not through optimization.

|Hyperparameters|CNN-Softmax|CNN-SVM|
|---------------|-----------|-------|
|Batch size|128|128|
|Learning rate|1e-3|1e-3|
|Steps|10000|10000|
|SVM C|N/A|1|


The experiments were conducted on a laptop computer with Intel Core(TM) i5-6300HQ CPU @ 2.30GHz x 4, 16GB of DDR3 RAM,
and NVIDIA GeForce GTX 960M 4GB DDR5 GPU.

![](figures/DNN_SVM on MNIST.jpg)

**Figure 1. Training accuracy (left) and loss (right) of CNN-Softmax and CNN-SVM on [MNIST](http://yann.lecun.com/exdb/mnist/).**

#### 2. CNN with SVM v.s. CNN with softmax on CIFAR_10




![](figures/accuracy-loss-fashion.png)

**Figure 2. Training accuracy (left) and loss (right) of CNN-Softmax and CNN-SVM on image classification using [Fashion-MNIST](http://github.com/zalandoresearch/fashion-mnist).**

The red plot refers to the training accuracy and loss of CNN-Softmax, with a test accuracy of 91.86000227928162%.
On the other hand, the light blue plot refers to the training accuracy and loss of CNN-SVM, with a test accuracy of
90.71999788284302%. The result on CNN-Softmax corroborates the finding by [zalandoresearch](https://github.com/zalandoresearch) on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist#benchmark).


