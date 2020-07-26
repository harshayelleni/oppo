# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 09:47:52 2020

@author: saiha
"""

import os
os.getcwd()
collection = "C:/Users/saiha/Downloads/OPPO/apppics_/" #C:/Users/saiha/Downloads/OPPO/noperson_resized_128_128/
for i, filename in enumerate(os.listdir(collection)):
    os.rename("C:/Users/saiha/Downloads/OPPO/apppics_/" + filename, "C:/Users/saiha/Downloads/OPPO/apppics_/" + str(i) + ".png")