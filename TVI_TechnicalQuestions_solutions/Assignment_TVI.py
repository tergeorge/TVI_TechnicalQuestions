#Teresa George 
# Load the required packages
import urllib.request
from PIL import Image
import numpy as np
import skimage.io

import matplotlib.pyplot as plt
from pathlib import Path

import os
script_dir = os.path.dirname(__file__)

############################################ Question 1######################################################

# Getting the file information 

import os

def get_file_information(filename,file_no):
    # Calculating the size of file in bytes
    print("The size of " + "Ch0" + str(file_no) + " file is " + str(os.path.getsize(filename))+ " bytes")
    # Getting other file details like st_mode the file type and permissions, st_ino the inode number,
    #st_dev the device id, st_uid the file owner id, st_gid the file group id and st_size the file size
    print(os.stat(filename))

    image = skimage.io.imread(fname=filename) # Reading the image
    image = np.asarray(image) # Converting into array
    l = image.shape[0] # Getting the shape
    b= image.shape[1]
    print("The Data Type of " + "Ch0" + str(file_no)+ " is %s" % image.dtype) # Getting the data type
    print('Min pixel value of file: %.3f , Max pixel value of file : %.3f' % (image.min(), image.max())) # Getting the minimum and maximum pixel value
    # converting from integers to floats
    image = image.astype('float32')
    # normalize the pixel value
    image /= 255.0
    # confirm the normalization
    print('Min normalized pixel value of file: %.3f , Max normalized pixel value of file : %.3f' % (image.min(), image.max())) # Get minimum and maximum of the normalized pixel value
    image_1D = image.reshape(image.shape[0]*image.shape[1]) # Convert to 1D array
    histogram_image, bin_edges = np.histogram(image_1D,bins=np.arange(0, 256)) # Get the histogram
    return histogram_image, bin_edges, image

# Ch01 file 

filename ="41077_120219_S000090_ch01.tif"
filename = os.path.join(script_dir, "TQ1", filename)
file_no = 1
histogram_image, bin_edges, image = get_file_information(filename,file_no)
# Plotting the histogram and the image
plt.figure(1)
plt.subplot(2,3,1)
plt.title(" Histogram image Ch01")
plt.xlabel("pixel value")
plt.ylabel("pixel count")
plt.yscale('log')
plt.plot(bin_edges[0:-1], histogram_image) 
plt.subplot(2,3,4)
plt.title("Image Ch01")
plt.imshow(image)


# File Ch02
filename ="41077_120219_S000090_ch02.tif"
filename = os.path.join(script_dir, "TQ1", filename)
file_no = 2
# Plotting the histogram and the image
histogram_image, bin_edges, image = get_file_information(filename,file_no)
plt.subplot(2,3,2)
plt.title(" Histogram image Ch02")
plt.xlabel("pixel value")
plt.ylabel("pixel count")
plt.yscale('log')
plt.plot(bin_edges[0:-1], histogram_image) 
plt.subplot(2,3,5)
plt.title("Image Ch02")
plt.imshow(image)

# File Ch03
filename ="41077_120219_S000090_ch03.tif"
filename = os.path.join(script_dir, "TQ1", filename)
file_no = 3
histogram_image, bin_edges, image = get_file_information(filename,file_no)
# Plotting the histogram and the image
plt.subplot(2,3,3)
plt.title(" Histogram image Ch03")
plt.xlabel("pixel value")
plt.ylabel("pixel count")
plt.yscale('log')
plt.plot(bin_edges[0:-1], histogram_image) 
plt.subplot(2,3,6)
plt.title("Image Ch03")
plt.imshow(image)
plt.show()




########################################### Question 2 ##############################################
import cv2
import numpy as np

# Load the Ground truth image
filename ="binary_41077_120219_S000090_L01.tif"
filename = os.path.join(script_dir, "TQ2", filename)
image_mask = skimage.io.imread(fname=filename)
image_mask = np.asarray(image_mask) # Convert it into an array
image_mask = image_mask.astype('float32') # Convert to float
image_mask /= 255.0 # Normalize the image

# To get the groud truth segmentation image from the images used in Q1
filename_3 ="41077_120219_S000090_ch03.tif"
filename_3 = os.path.join(script_dir, "TQ1", filename_3)
file_no_3 = 3
histogram_image, bin_edges, image_ch3 = get_file_information(filename_3,file_no_3)


filename_1 ="41077_120219_S000090_ch01.tif"
filename_1= os.path.join(script_dir, "TQ1", filename_1)
file_no_1 = 1
histogram_image, bin_edges, image_ch1 = get_file_information(filename_1,file_no_1)


image_new = image_ch3 - image_ch1 # Subtract image 1 from image 3
mask = cv2.inRange(image_new, 10, 255) # Threshold the image at 10 to get the segmentation mask

# apply morphology to remove isolated extraneous noise
# use borderconstant of black since foreground touches the edges
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Plot the generated segmentation mask with the Ground truth mask image to compare the two
plt.figure(2)
plt.subplot(1,2,1)
plt.title("Mask of image generated")
plt.imshow(mask)
plt.subplot(1,2,2)
plt.title("Ground truth mask")
plt.imshow(image_mask)
plt.show()

# Calculate the MSE metric value to comapre the two segmentation mask images 
from sklearn.metrics import mean_squared_error
print("MSE: ", mean_squared_error(mask,image_mask))

################################################ Question 3 ##########################################

# Compute the features from the segmentation mask computed in Q2
# initialize feature detector

orb = cv2.ORB_create()
orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
orb.setWTA_K(3)
    
# detect keypoints using the feature detector
kp = orb.detect(mask,None)

# Compute the descriptor values from the keypoints
kp, des = orb.compute(mask, kp)
# Draw the keypoints
img2_kp = cv2.drawKeypoints(mask, kp, None, color=(0,255,0), \
       flags=cv2.DrawMatchesFlags_DEFAULT)

# Plot the computed keypoints on the segmentation mask image
plt.figure(3)
plt.imshow(img2_kp)
plt.title("Key features")
plt.show()
# Print the computed descriptor values
print(des)

########################################### Question 4 #################################################
# To comapre the signals metric in the different images, three metrics are used : SNR of the individual images,
# Peak signal to noise ratio with respect to the mean signal value of all the images and the structural similarity index
# of each of the images with the mean of all the images

# Function to compute the signal to noise ration of the image
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def compute_metric(file_name):

    img  = skimage.io.imread (fname=file_name) # Reading each file individually
    img = img.astype('float32') # Converting to float 
    img /= 255.0
    img_shape=img.shape
    img = img.reshape(img.shape[0]*img.shape[1]) # Reshaping the image
    return img, img_shape

import scipy
ct = 0
mean_snr = np.empty(20, dtype=float) 
img_concat = np.array([])
filename_4 = os.path.join(script_dir, "TQ4","")
# Looping through all the images in the Q4 folder 
for i in range(71,90):
    file_name = filename_4 + "41077_120219_S0000" + str(i)+ "_ch03.tif"
    img, img_shape = compute_metric(file_name)
    # Concatenating all the individual images
    img_concat = np.vstack([img_concat, img]) if img_concat.size else img
    snr = signaltonoise(img) # Computing the signal to noise ratio
    mean_snr[ct] = np.mean(snr) # Computing the mean signal to noise ratio
    ct = ct+1
mean_img_concat = np.mean(img_concat,0) # Computing the mean of the conatenated image
mean_img_concat = mean_img_concat.reshape(img_shape[0],img_shape[1]) # Reshaping the concatenated image

from skimage.metrics import structural_similarity as ssim
ct = 0
mean_ssim = np.empty(20,dtype = float)
mean_psnr = np.empty(20, dtype=float) 
for i in range(71,90):
    file_name = filename_4 + "41077_120219_S0000" + str(i)+ "_ch03.tif"
    img, img_shape = compute_metric(file_name)
    img = img.reshape(img_shape[0],img_shape[1])
    # Computing the peak signal to noise ratio of each of the individual image with the mean PSNR of all the images
    psnr = cv2.PSNR(mean_img_concat, img)
    # Computing the structurl similarity index of the individual image with the mean SSIM of all the images 
    ssim_signal = ssim(mean_img_concat, img,data_range=img.max() - img.min())
    mean_psnr[ct] = np.mean(psnr) # Calculating the mean psnr
    mean_ssim[ct] = np.mean(ssim_signal) # Calculating the mean ssim 
    ct = ct+1

# Plotting the mean PSNR, mean SSIM and the SNR of all the individual images
plt.figure(4)
plt.subplot(1,3,1)
plt.title("Peak Signal to noise ratio")
plt.xlabel("Image number")
plt.ylabel("Mean PSNR value")
plt.plot(mean_psnr)

plt.subplot(1,3,2)
plt.title(" Mean structural similarity Index")
plt.xlabel("Image number")
plt.ylabel("Mean SSIM value")
plt.plot(mean_ssim)

plt.subplot(1,3,3)
plt.title("Signal to noise ratio of individual signals")
plt.xlabel("Image number")
plt.ylabel("Mean SNR")
plt.plot(mean_snr)
plt.show()
