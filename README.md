# Convolutional-Neural-Network-from-Scratch
Implementation of Convolutional Neural Networks on MNIST data set 

Check out the Live App @ http://madhav.pythonanywhere.com/



An **Optical and Handwritten digit recogniser**. A Convolution Neural Network trained over MNIST data set. 

A **Deep learning Model** made from scratch with only numpy. No other libraries/frameworks were used. 

### Motivation:
As part of my personal journey to gain a better understanding of Deep Learning, I’ve decided to build a Convolutional Neural Network from scratch without a deep learning library like TensorFlow. I believe that understanding the inner workings of a Neural Network is important to any aspiring Data Scientist.This allowed me to deeply understand every method in my model and gave me a better intution of Neural Networks.

**Screenshots** of the [live App](http://madhav.pythonanywhere.com/):

<img src="https://user-images.githubusercontent.com/27866638/55741277-86f82580-5a4a-11e9-98a8-abcc085a0b9f.png" width = "880">

<img src="https://user-images.githubusercontent.com/27866638/55739815-73978b00-5a47-11e9-8a81-f967ab9edf97.png" width = "440" height="550"><img src="https://user-images.githubusercontent.com/27866638/55740223-3da6d680-5a48-11e9-8614-d984f024afe3.png" width = "440" height="550">


## About MNIST dataset:
<img src="https://user-images.githubusercontent.com/27866638/55741644-68465e80-5a4b-11e9-87ef-e161e1fc499e.jpeg" width = "380">

The MNIST database of handwritten digits, available from [this page](http://yann.lecun.com/exdb/mnist/), has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field. you can also find dataset [here](https://github.com/madhavambati/Convolutional-Neural-Network-with-Numpy/tree/master/model).

## About Convolutional Neural Networks:
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics. The architecture of a ConvNet is analogous to that of the connectivity pattern of Neurons in the Human Brain and was inspired by the organization of the Visual Cortex. Individual neurons respond to stimuli only in a restricted region of the visual field known as the Receptive Field. A collection of such fields overlap to cover the entire visual area.

A typical CNN is made of the layers below:
- [Convolution Layer](https://www.youtube.com/watch?v=XuD4C8vJzEQ&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=2)
- [Pooling layer](https://www.youtube.com/watch?v=8oOgPUO-TBY&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=9)
- [Fully connecter layer](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/fc_layer.html)
- [Softmax layer (if necessary)](https://towardsdatascience.com/the-softmax-function-neural-net-outputs-as-probabilities-and-ensemble-classifiers-9bd94d75932)

Detailed description of all these layers can be found in the links given above. To Dive deep into Convolutional neural networks refer to the links given at the end of this readme 

# My ConvNet:
## Architecture:
