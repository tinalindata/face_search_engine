#!/usr/bin/python3
import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import meta_graph
from tqdm import tqdm

import face_detect_vectorize as fdv


MODEL_PATH = '20180402-114759/model-20180402-114759'
IMAGE_SZ = 160
model = fdv.Model(path=MODEL_PATH, image_size=IMAGE_SZ)

tasks = []
with open('../celebrity_barcodes_urls.csv', 'r') as f:
    for line in f:
        barcode = line.split(',')[0]
        tasks.append(barcode)
        pass
    pass

print("Load faceNet =>>>")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    model.loader(sess)
    print("Done!\n")
    for barcode in tqdm(tasks):
        save_dir = os.path.join('data', barcode[4:6], barcode[6:8], barcode)
        if os.path.exists(save_dir + '/features.npy'):
            continue
        try:
            with open(save_dir + '/info', 'rb') as f:
                info = pickle.load(f)
        except:
            continue
        L = len(info['faces'])
        if L > 0:
            try:
                batch = np.zeros((L, IMAGE_SZ, IMAGE_SZ, 3), dtype=np.uint8)
                for i in range(L):
                    image= cv2.imread('%s/%d.png' % (save_dir, i), cv2.IMREAD_COLOR)
                    batch[i, :, :, :] = cv2.resize(image, (IMAGE_SZ, IMAGE_SZ))
                    pass
                emb = sess.run(model.embeddings, feed_dict={model.images: batch})
                np.save(save_dir + '/features', emb)
            except:
                pass
        pass
