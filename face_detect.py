#!/usr/bin/python3
import os
import sys
import cv2
import urllib
import numpy as np
import face_detect_vectorize as fdv
from pyseeta import Detector

BARCODE_URLs_FILE = sys.argv[1]

tasks = []
with open(BARCODE_URLs_FILE, 'r') as IMAGES:
    for line in IMAGES:
        key, thumb, url = line.strip().split(',')
        tasks.append((key, url))

# log file for errors
logf = open(sys.argv[2], 'w')
logf.write("Loaded %d items\n" % len(tasks))

detector = Detector()
detector.set_min_face_size(30)
# detector.release()    <-- don't do it here.  do it when all processing is done.


def process(barcode, url):

    # Construct directory structure
    save_dir = os.path.join('data', barcode[4:6], barcode[6:8], barcode)
    try:
        os.makedirs(save_dir)
    except:
        if os.path.exists(os.path.join(save_dir, 'info')):
            logf.write("%s aleardy exist\n" % save_dir)
            return
        pass
    try:
        face = fdv.Face(image_path=url, output_dir=save_dir, is_url=True, key=barcode)
        face.detect_face(detector=detector)
        face.write_clip(write_to_file=True)
        face.write_info("info")
        logf.write("%s done\n" % save_dir)
    except Exception as e:
        print(e)
        os.system("mv %s %s_fail" % (save_dir, save_dir))
        logf.write("Failed to process %s (%s): %s\n" % (url, barcode, e))


for barcode, url in tasks:
    process(barcode, url)

logf.close()
