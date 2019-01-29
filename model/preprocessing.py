from scipy import ndimage
import math
import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

''' 
This file contains the preprocessing of image array that comes from the canvas by request method in flask '''

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


def preprocessing(img):

        img=255-np.array(img).reshape(28,28).astype(np.uint8)
        q, cimg = cv.threshold(img,127 , 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        cv.imshow('the_image', cimg)
        #cimg = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
           # cv.THRESH_BINARY,3,1)

        #cimg = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
           # cv.THRESH_BINARY,11,2)
        
        while np.sum(cimg[0]) == 0:  #making squared image with respective pixels
            cimg = cimg[1:]

        while np.sum(cimg[0,:]) == 0:
            cimg = cimg[:,1:]

        while np.sum(cimg[-1]) == 0:
            cimg = cimg[:-1]

        while np.sum(cimg[:, -1])==0:
            cimg = cimg[:,:-1]
            
        rows,cols = cimg.shape
        print(  "after shit",cimg.shape)
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
         
        ximg = centered.reshape(1,28,28).astype(np.float32)
        ximg-= int(33.3952)
        ximg/= int(78.6662)
        return ximg
