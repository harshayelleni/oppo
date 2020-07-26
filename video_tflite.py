# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 06:33:51 2020

@author: saiha
"""

import cv2, sys, time
import numpy as np
import tensorflow as tf
from PIL import Image

# Width and height
#width  = 320
#height = 240

# Frame rate
fps = ""
elapsedTime = 0

# Video capturer
cap = cv2.VideoCapture("C:/Users/saiha/Downloads/OPPO/videobokeh1.mp4") #videobokeh1
#guitars needs brightness augmentation
#docs
#discuss needs back side photos
cv2.namedWindow('FPS', cv2.WINDOW_NORMAL) #WINDOW_AUTOSIZE
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


# Initialize tflite-interpreter
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Downloads/OPPO/deeplab_v3_plus_mnv2_decoder_513.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Downloads/OPPO/deeplab_v3_plus_mnv2_decoder_513_latency.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Downloads/OPPO/deeplab_v3_plus_mnv2_decoder_513_size.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/transpose_seg/deconv_fin_munet_uint8.tflite") #models/transpose_seg/deconv_fin_munet.tflite
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Downloads/OPPO/pretrained/deeplab_aiseg_quant513.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/deeplab_dm05_513/deeplab_aiseg513_dm05_float.tflite") #can't reshape size 526338 into shape (513,513)
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Downloads/OPPO/pretrained/deeplabv3_mnv2_pascal_train_aug_8bit_2019_04_26/deeplabv3_mnv2_pascal_train_aug_8bit/deeplabv3_mnv2_pascal_train_aug_frozen_inference_graph.tflite")#C:\Users\saiha\Downloads\OPPO\pretrained\deeplabv3_mnv2_pascal_train_aug_8bit_2019_04_26\deeplabv3_mnv2_pascal_train_aug_8bit\
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/mnv3_seg/munet_mnv3_wm05.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/bilinear_seg/bilinear_fin_munet.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/prisma_seg/prisma-net.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/slim_seg_512/slim_reshape.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/segvid_mnv2_port256/portrait_video.tflite") #for video
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/prisma_seg/prisma-trinet.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/prisma_seg/prisma-net_fp16.tflite") #dequantize error (unsolved)
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/prisma_seg/prisma-net_uint8.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/prisma_seg/prisma_quant_edgetpu.tflite") #for edge tpu
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/mnv3_seg/munet_mnv3_wm10.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/gpucom/model1.tflite") #input shape mismatch(1x256x256x1 required input)
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/bilinear_seg/bilinear_fin_munet.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/bilinear_seg/bilinear_fin_munet_uint8.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Sites/Portrait-Segmentation/models/bilinear_seg/bilinear_fin_munet_fp16.tflite") #dequantize error
##interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Downloads/OPPO/trained/Aug3_up_super_fin_model-182-0.02.tflite")
##interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Downloads/OPPO/trained/PFAug3_up_super_fin_model-499-0.05.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Downloads/OPPO/trained/PFAug3_deconv_fin_model-421-0.08.tflite")
###interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Downloads/OPPO/trained/Aug1_up_super_fin_model-230-0.04.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Downloads/OPPO/trained/PFAug3_sam_deconv_fin_model-121-0.05.tflite")
#interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Downloads/OPPO/trained/PFAug4_samao_deconv_fin_model-808-0.09.tflite")
interpreter = tf.lite.Interpreter(model_path="C:/Users/saiha/Downloads/OPPO/trained/PFAug5_samao_deconv_fin_model-938-0.13.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]
print(input_details)
print(output_details)
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
# Image overlay
overlay = np.zeros((input_shape[0],input_shape[1], 3), np.uint8)
overlay[:] = (127, 0, 0)

#out = cv2.VideoWriter('output_munet_mnv3_wm05_f10.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('output_munet_mnv3_wm05_f20.mp4', fourcc, 20.0, (frame_width,frame_height))
while True:     
    # Read frames
    t1 = time.time()
    ret, frame = cap.read()
    # Display the resulting frame    
    #cv2.imshow('frame',frame)
    # Press Q on keyboard to stop recording
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #  break
    # Break the loop
    #else:
    #    break
    # BGR->RGB, CV2->PIL
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image=Image.fromarray(rgb)
    #t1 = time.time()
    #image = cv2.imread("C:/Users/saiha/Downloads/OPPO/input.jpg")    
    # Resize image
    image= image.resize(input_shape, Image.ANTIALIAS)

    t1 = time.time()
    #image = Image.open("C:/Users/saiha/Downloads/OPPO/input.jpg").resize((width, height))
    # Normalization
    image = np.asarray(image)
    prepimg = image / 255.0
    prepimg = prepimg[np.newaxis, :, :, :]

    # Segmentation
    interpreter.set_tensor(input_details[0]['index'], np.array(prepimg, dtype=np.float32)) #uint8
    interpreter.invoke()
    outputs = interpreter.get_tensor(output_details[0]['index'])
    elapsedTime = time.time() - t1
    #print(elapsedTime)
    print(outputs[0])
    # Process the output
    output = np.uint8(outputs[0]>0.55)
    #output = np.uint8(outputs[0]==1)
    res=np.reshape(output,input_shape)
    mask = Image.fromarray(np.uint8(res), mode="P")
    mask = np.array(mask.convert("RGB"))*overlay
    mask = cv2.resize(np.asarray(mask), (width,height),interpolation=cv2.INTER_CUBIC)
    frame = cv2.resize(frame, (width,height),interpolation=cv2.INTER_CUBIC)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    print(type(mask))
    print(mask.shape)    
    # Overlay the mask
    output = cv2.addWeighted(frame, 1, mask, 0.9, 0)
    #print(type(output))
    #print(output.shape)q
    #cv2.imwrite('C:/Users/saiha/Downloads/OPPO/maskop_dv3_513_aiseg.jpg', mask)
    #cv2.imshow("image", mask)
    #cv2.imwrite('C:/Users/saiha/Downloads/OPPO/imageop_dv3_513_aiseg.jpg', output)
    #elapsedTime = time.time() - t1
    #print(elapsedTime)
    #cv2.imshow("image", output)    
    # Display the output
    cv2.putText(output, fps, (width-180,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
    cv2.imshow('FPS', output)
    if ret == True:
        # Write the frame into the file 'output.avi'
        out.write(frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    else:
        break

    # Print frame rate
    fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)
    print("fps = ", str(fps))

cap.release()
out.release()
cv2.destroyAllWindows()

