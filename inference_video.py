# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:03:18 2020

@author: saiha
"""

import cv2, sys, time
import numpy as np
import tensorflow as tf
from PIL import Image

# Normalize the input image
def normalize(imgOri, scale=1, mean=[103.94, 116.78, 123.68], val=[0.017, 0.017, 0.017]):
    img = np.array(imgOri.copy(), np.float32)/scale
    return (img - mean) * val
 
# Alpha blend frame with background
def blend(frame, alpha):
        background = np.zeros(frame.shape) + [255, 255, 255]
        alphargb = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        result = np.uint8(frame * alphargb + background * (1-alphargb))
        return frame, alphargb*255, result


# Initialize tflite-interpreter
interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/segvid_mnv2_port256/portrait_video.tflite") # Use 'tf.lite' on recent tf versions
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]


# Initialize video capturer
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("C:/Users/saiha/Downloads/OPPO/girls.mp4") #5,1
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
cnt = 1

while True:
    
    # Read the BGR frames 
    ret, frame = cap.read()
    image=Image.fromarray(frame)
    
    # Resize the image
    image= image.resize((width, height), Image.ANTIALIAS)
    image=np.asarray(image)

    # Normalize the input
    image = normalize(image)

    # Choose prior mask
    if cnt == 1:
        prior = np.zeros((height, width, 1)) # first frame
    else:
        prior = pred_video
    
    # Add prior as fourth channel
    image=np.dstack([image,prior])
    prepimg = image[np.newaxis, :, :, :]
    

    # Invoke interpreter for inference
    interpreter.set_tensor(input_details[0]['index'], np.array(prepimg, dtype=np.float32))
    interpreter.invoke()
    outputs = interpreter.get_tensor(output_details[0]['index'])
    outputs = outputs.reshape(height,width,1)

    # Save output to feed subsequent inputs
    pred_video = outputs

    # Process the output
    outputs = cv2.resize(outputs, size)
    _,_,outputs=blend(frame, outputs)
    
    # Display the output
    cv2.imshow('Portrait Video',outputs)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
    
    # Print the frame count
    cnt += 1
    if cnt % 100 == 0:
        print ("cnt: ", cnt)

# When everything done, release the capturer
cap.release()
cv2.destroyAllWindows()

'''
Sample run: python portrait_video.py
'''