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

## Results

The hyperparameters used in this project were manually assigned, and not through optimization.

|Hyperparameters|CNN-Softmax|CNN-SVM|
|---------------|-----------|-------|
|Batch size|128|128|
|Learning rate|1e-3|1e-3|
|Steps|10000|10000|
|SVM C|N/A|1|

The experiments were conducted on a laptop computer with Intel Core(TM) i5-6300HQ CPU @ 2.30GHz x 4, 16GB of DDR3 RAM,
and NVIDIA GeForce GTX 960M 4GB DDR5 GPU.

![](figures/accuracy-loss-mnist.png)

**Figure 1. Training accuracy (left) and loss (right) of CNN-Softmax and CNN-SVM on image classification using
[MNIST](http://yann.lecun.com/exdb/mnist/).**

The orange plot refers to the training accuracy and loss of CNN-Softmax, with a test accuracy of 99.22999739646912%.
On the other hand, the blue plot refers to the training accuracy and loss of CNN-SVM, with a test accuracy of
99.04000163078308%. The results do not corroborate the findings of [Tang (2017)](https://arxiv.org/abs/1306.0239)
for [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/) classification. This may be attributed to the fact
that no data preprocessing nor dimensionality reduction was done on the dataset for this project.

![](figures/accuracy-loss-fashion.png)

**Figure 2. Training accuracy (left) and loss (right) of CNN-Softmax and CNN-SVM on image classification using [Fashion-MNIST](http://github.com/zalandoresearch/fashion-mnist).**

The red plot refers to the training accuracy and loss of CNN-Softmax, with a test accuracy of 91.86000227928162%.
On the other hand, the light blue plot refers to the training accuracy and loss of CNN-SVM, with a test accuracy of
90.71999788284302%. The result on CNN-Softmax corroborates the finding by [zalandoresearch](https://github.com/zalandoresearch) on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist#benchmark).

## Citation
To cite the paper, kindly use the following BibTex entry:
```
@article{agarap2017architecture,
  title={An Architecture Combining Convolutional Neural Network (CNN) and Support Vector Machine (SVM) for Image Classification},
  author={Agarap, Abien Fred},
  journal={arXiv preprint arXiv:1712.03541},
  year={2017}
}
```

To cite the repository/software, kindly use the following BibTex entry:
```
@misc{abien_fred_agarap_2017_1098369,
  author       = {Abien Fred Agarap},
  title        = {AFAgarap/cnn-svm v0.1.0-alpha},
  month        = dec,
  year         = 2017,
  doi          = {10.5281/zenodo.1098369},
  url          = {https://doi.org/10.5281/zenodo.1098369}
}
```

## License
```
Convolutional Neural Network with Support Vector Machine
Copyright (C) 2017-2020  Abien Fred Agarap

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
