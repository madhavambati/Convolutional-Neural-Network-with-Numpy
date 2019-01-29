
from functions import *
import numpy as np

def get_network_architecture(image, label, params, conv_stride, pooling_filter, pooling_stride):
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    # Forward Feeding
    # ***********------Forward feeding-------************

    convolution_1 = convolution(image, f1, b1, conv_stride)  # first covolution
    convolution_1[convolution_1 <= 0] = 0  # pass through ReLU non-linearity

    convolution_2 = convolution(convolution_1, f2, b2, conv_stride)  # second convolution
    convolution_2[convolution_2 <= 0] = 0  # pass through ReLU non-linearity

    maxpool_layer = maxpool(convolution_2, pooling_filter, pooling_stride)  # maxpooling

    (nf, dim, _) = maxpool_layer.shape

    fc = maxpool_layer.reshape((nf * dim * dim, 1))  # flattened layer

    z1 = w3.dot(fc) + b3  # dense layer_1
    z1[z1 <= 0] = 0  # ReLU non-linearity

    out = w4.dot(z1) + b4  # dense layer_2

    pred = softmax(out)  # pass through softmax function

    # loss estimation
    # '***********------Calculating Loss-------************'

    loss = loss_function(pred, label)

    # Backing Propagation
    # '***********------Backpropagation-------************'

    dout = pred - label  # derivative of loss w.r.t. final dense layer output
    dw4 = dout.dot(z1.T)  # loss gradient of final dense layer weights
    db4 = np.sum(dout, axis=1).reshape(b4.shape)  # loss gradient of final dense layer biases

    dz1 = w4.T.dot(dout)
    dz1[z1 <= 0] = 0
    dw3 = dz1.dot(fc.T)
    db3 = np.sum(dz1, axis=1).reshape(b3.shape)

    dfc = w3.T.dot(dz1)
    dmaxpool_layer = dfc.reshape(maxpool_layer.shape)

    dconvolution_2 = maxpool_backprop(dmaxpool_layer, convolution_2, pooling_filter, pooling_stride)
    dconvolution_2[convolution_2 <= 0] = 0

    dconvolution_1, df2, db2 = convolution_backprop(dconvolution_2, convolution_1, f2, conv_stride)
    dconvolution_1[convolution_1 <= 0] = 0

    dimage, df1, db1 = convolution_backprop(dconvolution_1, image, f1, conv_stride)

    gradients = [df1, df2, dw3, dw4, b1, b2, b3, b4]
    return gradients, loss


# In[23]:


# Adams optimizer is used to optimise the cost function
# Adam is an adaptive learning rate optimization algorithm
# that’s been designed specifically for training deep neural networks
# Adams optimizer is explained in the link below
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

def optimize_network(batch, num_classes, alpha, dim, n_c, beta1, beta2, params, cost_array, E=1e-7):
    # print(E)
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    batch_size = len(batch)

    images = batch[:, 0:-1]  # take upto n-1 columns in the data set, last column in the test set will be labels
    images = images.reshape((batch_size, n_c, dim, dim))  # reshape the 784(28*28) columns to feed into network

    labels = batch[:, -1]  # last column in the data set is labels

    cost = 0

    # initialise gradients
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    dw3 = np.zeros(w3.shape)
    dw4 = np.zeros(w4.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)

    # initialise momentum params
    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    vb1 = np.zeros(b1.shape)
    vb2 = np.zeros(b2.shape)
    vb3 = np.zeros(b3.shape)
    vb4 = np.zeros(b4.shape)

    # initialise RMS-prop params
    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    sb1 = np.zeros(b1.shape)
    sb2 = np.zeros(b2.shape)
    sb3 = np.zeros(b3.shape)
    sb4 = np.zeros(b4.shape)

    for i in range(batch_size):
        x = images[i]

        # one-hot encoding of labels to avoid any numerical relationship between labels
        y = np.eye(num_classes)[int(labels[i])].reshape((num_classes, 1))

        gradients, loss = get_network_architecture(x, y, params, 1, 2, 2)

        [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = gradients

        df1 += df1_
        df2 += df2_
        dw3 += dw3_
        dw4 += dw4_
        db1 += db1_
        db2 += db2_
        db3 += db3_
        db4 += db4_

        cost += loss

    # updating momentum and RMS-prop parameters
    v1 = beta1 * v1 + (1 - beta1) * df1 / batch_size
    s1 = beta2 * s1 + (1 - beta2) * (df1 / batch_size) ** 2
    f1 -= alpha * v1 / np.sqrt(s1 + E)

    v2 = beta1 * v2 + (1 - beta1) * df2 / batch_size
    s2 = beta2 * s2 + (1 - beta2) * (df2 / batch_size) ** 2
    f2 -= alpha * v2 / np.sqrt(s2 + E)

    v3 = beta1 * v3 + (1 - beta1) * dw3 / batch_size
    s3 = beta2 * s3 + (1 - beta2) * (dw3 / batch_size) ** 2
    w3 -= alpha * v3 / np.sqrt(s3 + E)

    v4 = beta1 * v4 + (1 - beta1) * dw4 / batch_size
    s4 = beta2 * s4 + (1 - beta2) * (dw4 / batch_size) ** 2
    w4 -= alpha * v4 / np.sqrt(s4 + E)

    vb1 = beta1 * vb1 + (1 - beta1) * db1 / batch_size
    sb1 = beta2 * sb1 + (1 - beta2) * (db1 / batch_size) ** 2
    b1 -= alpha * vb1 / np.sqrt(sb1 + E)

    vb2 = beta1 * vb2 + (1 - beta1) * db2 / batch_size
    sb2 = beta2 * sb2 + (1 - beta2) * (db2 / batch_size) ** 2
    b2 -= alpha * vb2 / np.sqrt(sb2 + E)

    vb3 = beta1 * vb3 + (1 - beta1) * db3 / batch_size
    sb3 = beta2 * sb3 + (1 - beta2) * (db3 / batch_size) ** 2
    b3 -= alpha * vb3 / np.sqrt(sb3 + E)

    vb4 = beta1 * vb4 + (1 - beta1) * db4 / batch_size
    sb4 = beta2 * sb4 + (1 - beta2) * (db4 / batch_size) ** 2
    b4 -= alpha * vb4 / np.sqrt(sb4 + E)

    cost = cost / batch_size
    cost_array.append(cost)

    params = [f1, f2, w3, w4, b1, b2, b3, b4]

    return params, cost_array