#!/usr/bin/env python3

import os
import sys
import fawn
import numpy as np
import face_detect_vectorize as fdv
from tqdm import tqdm

BARCODE_URLs_FILE = sys.argv[1]


# extract features and messages of photos in the picture database
def extract_features(barcode):
    faces = {}
    path = os.path.join('data', barcode[4:6], barcode[6:8], barcode)
    info = fdv.read_pickle(path, "info")
    if len(info['faces']) == 0:
        return faces
    else:
        features = np.load(path + '/features.npy')
        for face in info['faces']:
            faces[face['ID']] = features[int(face['ID'])]
        return faces


# log file for errors
logf = open(sys.argv[2], 'w')

# The server url of image database
client = fawn.Fawn('http://127.0.0.1:8888')

# insert the image information into the database
tasks = []
with open(BARCODE_URLs_FILE, 'r') as IMAGES:
    for line in IMAGES:
        key, thumb, _ = line.strip().split(',')
        tasks.append((key, thumb))

for task in tqdm(tasks):
    try:
        faceFeatures = extract_features(barcode=task[0])
        thumbnail = task[1]
    except Exception as ex:
        logf.write('%s: %s \n' % (barcode, ex))

    if faceFeatures == {}:
        continue
    else:
        for faceId, faceVector in faceFeatures.items():
            try:
                # client.insert(key, vector, meta)
                # key is the path to reach the faces
                # meta is the thumbnail of the origin figure
                client.insert('%s/%s/%s/%s/%s' % ("data", barcode[4:6], barcode[6:8], barcode, faceId), faceVector.reshape(512), thumbnail)
            except Exception as e:
                logf.write("%s/%s/%s/%s/%s: %s\n" % ("data", barcode[4:6], barcode[6:8], barcode, faceId, e))

client.sync()

logf.close()

