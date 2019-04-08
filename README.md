# Convolutional Neural Network from Scratch
Check out the **Live App** @ http://madhav.pythonanywhere.com/

Implementation of Convolutional Neural Networks on MNIST dataset.

Achieved an **accuracy score of 97%** on MNIST dataset.



An **Optical and Handwritten digit recogniser**.

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


         INPUT -> ConvLayer1 -> ConvLayer2 -> MAXPOOL -> FC -> DenseLayer -> OutputLayer -> Softmax Activation


<img src="https://user-images.githubusercontent.com/27866638/55744651-14d80e80-5a53-11e9-8597-0b5601de0b96.png">

Image transition after each layer through the Network.

- Input Image - 28×28×1
- ConvLayer1  - 24×24×8
- ConvLayer2  - 20×20×8
- MaxpoolLayer- 10×10×8
- FC Layer    - 800×1
- DenseLayer  - 128×1
- OutputLayer - 10×1

**Accuracy: 97%**

## Training the Network:
Initially the weights are set to random. To make for a smoother training process, we initialize each filter with a mean of 0 and a standard deviation of 1. [Batch Normalisation](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c) into 32 batches.Batch normalization reduces the amount by what the hidden unit values shift around (covariance shift) and Labels are [one-hot encoded](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) to avoid any numerical relationships between the other labels. During Forward Feed [RELU](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0) non-linearity is used at every layer, loss has been calculated. The gradients for each layer are defined. [Adams optimizer](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) is used to optimise the cost function. Adam is an adaptive learning rate optimization algorithm that’s been designed specifically for training deep neural networks. A better explanation of Adam found [here](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/). It took 6hrs to train the network on my Intel i7 4600hq processor.

Some of the images during training:

<img src="https://user-images.githubusercontent.com/27866638/55750580-39d37e00-5a61-11e9-9804-ac31dd5366b2.png" width="880">

<img src="https://user-images.githubusercontent.com/27866638/55751473-5f618700-5a63-11e9-973d-75662fc07b61.png" width="880">


## Testing the Network:
After the CNN has finished training, a .pkl file containing the network’s parameters is saved to the directory where the script was run.

Network is tested using the trained parameters to run predictions on all 10,000 digits in the test dataset. After all predictions are made
an accuracy score of **97.3%** has been achieved.

## Web Application:
An interactive canvas was created when the the predict button is clicked the image data is sent as a jason string and passed through a prediction algorithm. The predicted data/number is displayed at the bottom of the canvas. you can also see the prediction probability in your browser console.

<img src="https://user-images.githubusercontent.com/27866638/55752974-e106e400-5a66-11e9-9898-a46d9c777bd3.png" width="880">

## Installation:
Use the following commands to install the model in your machine. The network is already trained and the parameters are saved in [params.pkl](https://github.com/madhavambati/Convolutional-Neural-Network-with-Numpy/blob/master/model/params.pkl) file. You can train the network yourself or you can use it by running [predict.py](https://github.com/madhavambati/Convolutional-Neural-Network-with-Numpy/blob/master/model/predict.py) file, don't forget to save your testing image in [model_images](https://github.com/madhavambati/Convolutional-Neural-Network-with-Numpy/tree/master/images_model) directory.

- Clone the repository

      git clone https://github.com/madhavambati/Convolutional-Neural-Network-with-Numpy.git
      

- Moveto directory Convolutional-Neural-Network-with-Numpy

      cd Convolutional-Neural-Network-with-Numpy
      
- First install all the dependencies 
     
      pip install -r requirements.text

      
- If you like to train the network yourself. But it took a solid 5hrs for me to train the network.
      
      cd model
      
      python train.py
      
  
- To check the accuracy on MNIST dataset

      python test.py 
      
- To predict a random number from an image, save the image in model_images directory and open the file predict.py and change the path

      python predict.py
     
       
