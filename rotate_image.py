# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:32:33 2020

@author: saiha
"""
'''
import cv2, sys, time
import numpy as np
img = cv2.imread("C:/Users/saiha/Downloads/OPPO/input.jpg") #LoadImage path_to_image.jpg
timg = cv2.CreateImage((img.height,img.width), img.depth, img.channels) # transposed image

# rotate counter-clockwise
cv2.Transpose(img,timg)
cv2.Flip(timg,timg,flipMode=0)
cv2.SaveImage("C:/Users/saiha/Downloads/OPPO/input_rot_ac.jpg", timg) #rotated_counter_clockwise.jpg

# rotate clockwise
cv2.Transpose(img,timg)
cv2.Flip(timg,timg,flipMode=1)
cv2.SaveImage("C:/Users/saiha/Downloads/OPPO/input_rot_c.jpg", timg) #rotated_clockwise.jpg
'''

import cv2

img=cv2.imread("C:/Users/saiha/Downloads/OPPO/input.jpg")

# rotate ccw
out=cv2.transpose(img)
out=cv2.flip(out,flipCode=0)

# rotate cw
out=cv2.transpose(img)
out=cv2.flip(out,flipCode=1)

cv2.imwrite("input_rotated.jpg", out)