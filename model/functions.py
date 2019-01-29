import numpy as np
import gzip

''' 
This file contains all the essential functions that are used in the network '''


''' Getting data '''
#Extract images by reading the file bytestream.
#Reshape the read values into a 2D matrix of dimensions [n, h*w]

def extract_data(filename, num_images, IMAGE_WIDTH):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data

#Extract labels by reading the file bytestream.
#Reshape the read values into a row matrix of dimensions [n, 1]

def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return labels




''' Initialising weights and biases for all the layers'''
#random values for filters in convolution layers
def Filter_weights(size):
    #Initialize filter using a normal distribution with and a
    #standard deviation inversely proportional the square root of the number of units
    stddev = 1/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0.0, scale = stddev, size = size)
#random values for weights in deep layers

def deep_weights(size):
    #Initialize weights with a random normal distribution
    return np.random.standard_normal(size = size)*0.01


'''convolution function'''

def convolution(image, Filter, bias, stride=1):
    # convolution of input image with a filter of dimensions(n_f,n_c,f,f)
    # n_f is no.of filters
    # n_c is no.of channels
    # f,f are height & width

    # image dimensions(n_c, image_h, image_w)
    # n_c is no.of channels in image
    # img_h is height of image
    # img_w is width of image

    (n_c, img_h, img_w) = image.shape
    (n_f, n_c, f, f) = Filter.shape

    # output dimensions after convolution
    out_h = int((img_h - f) / stride) + 1  # height of output matrix
    out_w = int((img_h - f) / stride) + 1  # width of output matrix
    # n_f will be the depth of the matrix

    out = np.zeros((n_f, out_h, out_w))

    # convolution of image_array with filter yeilds out_array
    # for i in range of no.of filters
    # define a row , out_y variabless to hover along rows of image, out_matrix respectively
    # define a column , out_x variables to hover along columns of image, out_matrix respectively
    # convolution is done in the ranges of image_height to image_width
    for i in range(n_f):
        row = out_row = 0

        while row + f <= img_h:

            column = out_column = 0

            while column + f <= img_w:
                out[i, out_row, out_column] = np.sum(Filter[i] * image[:, row: row + f, column: column + f]) + bias[i]
                column += stride
                out_column += 1

            row += stride
            out_row += 1

    return out


'''Maxpooling function'''

def maxpool(image, f=5, stride=2):
    (n_c, img_h, img_w) = image.shape  # input image dimension
    out_h = int((img_h - f) / stride) + 1  # output image height
    out_w = int((img_w - f) / stride) + 1  # output image width
    max_out = np.zeros((n_c, out_h, out_w))  # matrix to hold maxpooled image(or)array

    # maxpool of image_array with filter yeilds max_out array
    # for i in range of no.of channels
    # define a row , out_y variables to hover along rows of image, out_matrix respectively
    # define a column , out_x variables to hover along columns of image, out_matrix respectively

    for i in range(n_c):
        row = out_row = 0

        while row + f <= img_h:  # slide the max pooling window vertically(along rows) across the image

            column = out_column = 0

            while column + f <= img_w:  # slide the max pooling window vertically(along columns) across the image
                # choose the maximum value within the window at each step and store it to the output matrix
                max_out[i, out_row, out_column] = np.max(image[i, row: row + f, column: column + f])
                column += stride
                out_column += 1

            row += stride
            out_row += 1

    return max_out



'''Softmax function'''

def softmax(activations):
    # activations raised to the power of 'e'
    activations_raised_exp = np.exp(activations)

    # divide by sum of all exponentiated activations to get the required probability (0,1)
    probabilities = activations_raised_exp / np.sum(activations_raised_exp)
    return probabilities

'''Loss function'''

def loss_function(pred, label):
    # loss function for softmaxlayer will be -Σylogŷ
    # where- y is given label and ŷ is pred output
    net_loss = -np.sum(label * np.log(pred))
    return net_loss



'''Convolution during backpropagation'''

# back-propagation operations in convolution layers
# for convolution_backward we need derivative of convolution in the previous layer
# 'dconv_prev' is the derivative of convolution in the previous layer
# 'image' is referred to as the input for the current conv layer with which the convolution operation is applied
# 'Filter' is the filter used in the current layer
# The obtained convoluted matrix will be the 'conv_prev' for the current layer in backpropagation
# Backpropagation of the convolution layers is explained in the link below
# https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e

def convolution_backprop(dconv_prev, image, Filter, stride):
    (n_f, n_c, f, f) = Filter.shape
    (n_c, img_h, img_w) = image.shape

    dimage = np.zeros(image.shape)
    dFilter = np.zeros(Filter.shape)
    dbias = np.zeros((n_f, 1))
    for i in range(n_f):
        row = dimage_y = 0
        while row + f <= img_h:
            column = dimage_x = 0
            while column + f <= img_w:
                dFilter[i] += dconv_prev[i, dimage_y, dimage_x] * image[:, row:row + f, column:column + f]
                dimage[:, row:row + f, column:column + f] += dconv_prev[i, dimage_y, dimage_x] * Filter[i]
                column += stride
                dimage_x += 1
            row += stride
            dimage_y += 1
        dbias[i] = np.sum(dconv_prev[i])
    return dimage, dFilter, dbias


'''Maxpooling during backpropagation'''


# back-propagation in maxpool layer
# 'dpooled' is the derivative of previous layer i.e pooled layer
# 'maxpooled' is the maxpool layer's output
# 'Filter' will be 2*2 with 'stride' = 2
# save the index of the input image at which the max values are captured in the maxpool layer
# use the index to iterate across the output matrix while equating the corresponding higher values in pooledlayer
# back propagation in maxpool layers are explained in the link below
# https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/pooling_layer.html

def maxpool_backprop(dpooled, maxpooled, Filter, stride):
    (n_c, maxpooled_dim, _) = maxpooled.shape  # maxpooled_height = maxpooled_width=maxpooled_dim
    dmaxpooled = np.zeros(maxpooled.shape)

    for i in range(n_c):
        row = dmaxpooled_y = 0
        while row + Filter <= maxpooled_dim:
            column = dmaxpooled_x = 0
            while column + Filter <= maxpooled_dim:
                # obtain index of largest value in input for current window
                index = np.nanargmax(maxpooled[i, row:row + Filter, column:column + Filter])
                (a, b) = np.unravel_index(index, maxpooled[i, row:row + Filter, column:column + Filter].shape)
                dmaxpooled[i, row + a, column + b] = dpooled[i, dmaxpooled_y, dmaxpooled_x]

                column += stride
                dmaxpooled_x += 1
            row += stride
            dmaxpooled_y += 1

    return dmaxpooled

'''predict function'''
# after training the neural net just do the Forward-feed

def predict(image, params, conv_stride = 1, pooling_filter = 2, pooling_stride = 2 ):
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    print('done')
    convolution_1 = convolution(image, f1, b1, conv_stride)  # first covolution
    convolution_1[convolution_1 <= 0] = 0  # pass through ReLU non-linearity

    convolution_2 = convolution(convolution_1, f2, b2, conv_stride)  # second convolution
    convolution_2[convolution_2 <= 0] = 0  # pass through ReLU non-linearity

    maxpool_layer = maxpool(convolution_2, pooling_filter, pooling_stride)  # maxpooling

    (nf, dim, _) = maxpool_layer.shape
    print('done')
    fc = maxpool_layer.reshape((nf * dim * dim, 1))  # flattened layer
    print(fc.shape)
    z1 = w3.dot(fc) + b3  # dense layer_1
    z1[z1 <= 0] = 0  # ReLU non-linearity

    out = w4.dot(z1) + b4  # dense layer_2

    probabilities = softmax(out)  # pass through softmax function

    pred = np.argmax(probabilities)
    prob = np.max(probabilities)
    
    #prob = max(probabilities)

    #for i in range(10):
     #   if(probabilities[i] == max(probabilities)):
      #      pred = i

    

    return pred, prob



