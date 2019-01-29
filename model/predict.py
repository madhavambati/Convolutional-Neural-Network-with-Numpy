
from functions import *
from PIL import Image, ImageFilter
from scipy import ndimage
import pickle as pickle
import matplotlib.pyplot as plt
import cv2 as cv
import math
import os
from PIL import Image


if __name__ == '__main__':

    #im = Image.open('images/index.png')
    #rgb_im = im.convert('RGB')
    #rgb_im.save('images/96.jpg')

    
    def get_center_of_mass(img):
        Y,X = ndimage.measurements.center_of_mass(img)
        x,y = img.shape
        delta_x = np.round(y/2.0-X).astype(int)
        delta_y = np.round(x/2.0-Y).astype(int)
        return delta_x, delta_y
    
    def get_to_center(image ,x, y):
        
        (rows , cols) = image.shape
        M = np.float32([[1,0,x],[0,1,y]])
        centered = cv.warpAffine(image,M,(cols,rows))
        return centered 
    
    
    def preprocessing(cimg):
        
        
        q, cimg = cv.threshold(gray,127 , 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        cv.imshow('the_image', cimg)
        #cimg = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
           # cv.THRESH_BINARY,3,1)
        
        #cimg = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
           # cv.THRESH_BINARY,11,2)
        
        while np.sum(cimg[0]) == 0:
            cimg = cimg[1:]

        while np.sum(cimg[0,:]) == 0:
            cimg = cimg[:,1:]

        while np.sum(cimg[-1]) == 0:
            cimg = cimg[:-1]

        while np.sum(cimg[:, -1])==0:
            cimg = cimg[:,:-1]
            
        rows,cols = cimg.shape
        
        if rows == cols:
            nrows = 20
            ncols = 20
            cimg = cv.resize(cimg, (ncols,nrows))
            
        
        if rows > cols:
            nrows = 20
            ncols = int(round((cols*20.0/rows), 0))
            cimg = cv.resize(cimg, (ncols,nrows))
            
        
        else:
            ncols = 20
            nrows = int(round((rows*20.0/cols), 0))
          
            cimg = cv.resize(cimg, (ncols,nrows))
            
                             
        print(nrows, ncols)
        col_pad = (int(math.ceil((28-ncols)/2.0)), int(math.floor((28-ncols)/2.0)))
        
        row_pad = (int(math.ceil((28-nrows)/2.0)), int(math.floor((28-nrows)/2.0)))
        cimg = np.lib.pad(cimg,(row_pad,col_pad),'constant')
        print(cimg.shape)
        
        del_x, del_y = get_center_of_mass(cimg) 
        centered = get_to_center(cimg ,del_x, del_y)
         
        img = centered.reshape(1,28,28).astype(np.float32)
        img-= int(33.3952)
        img/= int(78.6662)
        return img
        
    print(os.getcwd())
    img = cv.imread('images_model/index1.png', -1 )
    
    
    cv.imshow('image',img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    print(gray.shape)
    print(len(gray[gray!=0]))
    gray = 255 - np.array(gray).astype(np.uint8)
    print(gray.shape)
    
    
    processed_img = preprocessing(gray)
    save_path = 'params.pkl'
    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
    label, prob = predict(processed_img, params)
    print(label)
    print("%0.2f"%prob)
    a =1 
    print(type(a))
    images_repr = processed_img.reshape(processed_img.shape[0], 28, 28)
    plt.imshow(images_repr[0], cmap=plt.get_cmap('gray'))
    plt.show()




