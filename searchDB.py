import os
import cv2
import sys
import fawn
import tensorflow as tf
import numpy as np
import face_detect_vectorize as fdv
from pyseeta import Detector

IMAGE_SZ = 160
MODEL_PATH = '20180402-114759/model-20180402-114759'
client = fawn.Fawn('http://127.0.0.1:8888')
SAMPLE = sys.argv[1]

model = fdv.Model(path=MODEL_PATH, image_size=IMAGE_SZ)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print("Load faceNet =>>>")
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    model.loader(sess)
    print("Done!\n")
  
    detector = Detector()
    detector.set_min_face_size(30)

    face = fdv.Face(SAMPLE, None)
    face.detect_face(detector=detector)
    face.write_clip(write_to_file = False)
    face.vectorize(sess=sess, model=model, model_input_size=(IMAGE_SZ, IMAGE_SZ))

    for feature in face.face_features: 
        res = client.search(feature.reshape(512), K=5)
        print(res)
        print("-----------")


