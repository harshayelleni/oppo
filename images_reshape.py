# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 08:39:21 2020

@author: saiha
"""

#!/usr/bin/python
from PIL import Image
import os, sys

path = "/media/data/sanjay/person-segmentation/masks/" #"C:/Users/saiha/Downloads/OPPO/apppics/" #/root/Desktop/python/images/
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((224,224), Image.ANTIALIAS)
            imResize.save(f + '_resized.png', 'PNG', quality=90) #JPEG

resize()
