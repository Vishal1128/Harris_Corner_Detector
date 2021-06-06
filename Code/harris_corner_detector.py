#!/usr/bin/env python
# coding: utf-8

# In[18]:


## Import required packages
import os
import sys
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.filters import convolve

get_ipython().run_line_magic('matplotlib', 'inline')

def load_img(img_path):
    img = mpimg.imread(img_path).astype('float64')/255.0
    luminence = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    return luminence

def get_img_grad(img):
    sobel_x = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]], np.float64) 
    sobel_y = np.asarray([[1,2,1],[0,0,0],[-1,-2,-1]], np.float64) 
    Fx = ndimage.filters.convolve(img, sobel_x)
    Fy = ndimage.filters.convolve(img, sobel_y)
    return Fx, Fy

def get_Rmatrix(img, Ix, Iy, m = 4, k = 0.04):
    Ixx = Ix**2
    Ixy = Iy*Ix
    Iyy = Iy**2
    R_values = np.zeros_like(img, dtype=np.float64)
    A, B = img.shape
    for i in range(m, A-m):
        for j in range(m, B-m):
            Kxx = Ixx[i-m : i+m+1, j-m : j+m+1]
            Kxy = Ixy[i-m : i+m+1, j-m : j+m+1]
            Kyy = Iyy[i-m : i+m+1, j-m : j+m+1]
            Sxx = Kxx.sum()
            Sxy = Kxy.sum()
            Syy = Kyy.sum()
            determinant = Sxx*Syy - Sxy**2
            trace = Sxx + Syy
            r_value = determinant - k* (trace**2)
            R_values[i][j] = r_value
    return R_values

def get_corner_list_from_Rvalues(img, R_values, ratio=0.25):
    corner_list = []
    T = ratio*R_values.max()
    M, N = img.shape
    for i in range(M):
        for j in range(N):
            if R_values[i][j] > T:
                corner_list.append([R_values[i][j], i, j])
    return corner_list 

def get_indices_connected_neighbours(corner_list, x, y):
    indices = []
    for i in range(len(corner_list)):
        if corner_list[i][1] != -1:
            if abs(corner_list[i][1] - x) <= 3 and abs(corner_list[i][2] - y) <= 3:
                indices.append(i)
    return indices

def non_maximal_suppression(corner_list):
    final_corner_list = []
    sorted_ind = np.argsort(np.array(corner_list)[:,0])
    for i in range(len(sorted_ind)):
        if corner_list[sorted_ind[i]][1] != -1:
            x = corner_list[sorted_ind[i]][1]
            y = corner_list[sorted_ind[i]][2]
            final_corner_list.append([corner_list[sorted_ind[i]][0], x, y])
            corner_list[sorted_ind[i]][1] = -1
            corner_list[sorted_ind[i]][2] = -1
            ind_list = get_indices_connected_neighbours(corner_list, x, y)
            for j in range(len(ind_list)):
                corner_list[ind_list[j]][1] = -1
                corner_list[ind_list[j]][2] = -1
    return final_corner_list

def get_corner_img(image, L):
    out = image.copy()
    for a in range(len(L)):
        x = int(L[a][1])
        y = int(L[a][2])
        out[x][y][0] = 1.0
        out[x][y][1] = 0.0
        out[x][y][2] = 0.0
        for i in range(x-3, x+4):
            out[i][y][0] = 1.0
            out[i][y][1] = 0.0
            out[i][y][2] = 0.0
        for i in range(y-3, y+4):
            out[x][i][0] = 1.0
            out[x][i][1] = 0.0
            out[x][i][2] = 0.0  
    return out

ROOT_DIR = os.path.dirname(os.getcwd())
print('Root Directory : ', ROOT_DIR, '\n')
IMAGES_DIR = os.path.join('/', ROOT_DIR, 'Data')
print('Images Directory : ', IMAGES_DIR, '\n')


m = 3
k = 0.04
threshold_ratio = 0.1

images = os.listdir(IMAGES_DIR)

for image in images:
    img_path = os.path.join('/', IMAGES_DIR, image)
    img = load_img(img_path)

    plt.figure()
    plt.title('Luminence Image')
    plt.imshow(img, cmap=plt.get_cmap('gray'))

    Ix, Iy = get_img_grad(img)
    R_values = get_Rmatrix(img, Ix, Iy, m, k)
    L = get_corner_list_from_Rvalues(img, R_values, threshold_ratio)

    plt.figure()
    img = mpimg.imread(img_path).astype('float64')/255.0
    plt.imshow(img)
    plt.show()


    plt.figure()
    plt.title('Threshold Image')
    before_NMS = get_corner_img(img, L)
    plt.imshow(before_NMS)
    plt.show()
    
    L_final = non_maximal_suppression(np.copy(L))
    img = mpimg.imread(img_path).astype('float64')/255.0

    plt.figure()
    plt.title('Harris Edge detected ')
    harris_output = get_corner_img(img, L_final)
    plt.imshow(harris_output)
    plt.show()
    
    print('Harris Corner Detection done for ' + image)
    print('Number of Corners Detected ' + str(len(L_final)), '\n')


# In[ ]:




