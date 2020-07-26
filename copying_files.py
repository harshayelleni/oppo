# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 10:35:53 2020

@author: saiha
"""

import shutil

for i in range(299):
    shutil.copy2('C:/Users/saiha/Downloads/OPPO/mask_camdata/zmask.png', 'C:/Users/saiha/Downloads/OPPO/mask_camdata/zmask{}.png'.format(i))