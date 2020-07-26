# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 09:32:33 2020

@author: saiha
"""

import cv2
import numpy as np
img = np.zeros([128,128,3],dtype=np.uint8)
img.fill(0)
from PIL import Image
im = Image.fromarray(img)
im.save("zmask.png")