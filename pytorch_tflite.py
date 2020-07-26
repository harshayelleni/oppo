# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 22:51:56 2020

@author: saiha
"""

import torch
import torchvision
import model_mobilenetv2_seg_small

dummy_input = torch.randn(1, 4, 224, 224).cuda()
model =  torch.load('/content/pnet_video.pth').cuda()

torch.onnx.export(model, dummy_input, "pnet_video.onnx", verbose=True)
'''
!pip install tensorflow-gpu
!pip install onnx
!git clone https://github.com/nerox8664/onnx2keras.git
'''
#%cd onnx2keras


import tensorflow as tf
import onnx
from onnx2keras import onnx_to_keras
from tensorflow.keras.models import load_model

# Load ONNX model
onnx_model = onnx.load('/content/pnet_video.onnx')

# Call the converter and save keras model
k_model = onnx_to_keras(onnx_model, ['input.1'],change_ordering=True)
k_model.save('/content/pnet_video.h5')

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation, Lambda, Reshape

# Load keras model
k_model=load_model('/content/pnet_video.h5')
k_model.summary()

# Remove edge branch from output
edge_model=Model(inputs=k_model.input,outputs=k_model.layers[-2].output)
edge_model.summary()

# Add softmax on output
sm=Lambda(lambda x: tf.nn.softmax(x))(edge_model.output)
soft_model=Model(inputs=edge_model.input, outputs=sm)
soft_model.summary()

# Get foreground softmax slice
ip = soft_model.output
str_slice=Lambda(lambda x: tf.strided_slice(x, [0,0, 0, 1], [1,224, 224, 2], [1, 1, 1, 1]))(ip)
stride_model=Model(inputs=soft_model.input, outputs=str_slice)
stride_model.summary()

# Flatten output
output = stride_model.output
newout=Reshape((50176,))(output)
reshape_model=Model(stride_model.input,newout)
reshape_model.summary()

# Save keras model
reshape_model.save('/content/portrait_video.h5')

# Convert to tflite
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(reshape_model)
tflite_model = converter.convert()
open("/content/portrait_video.tflite", "wb").write(tflite_model)