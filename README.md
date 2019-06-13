# Face Search Engine

Authors: Tina Lin, Kevin Deng, Mark Wu 

## Demo 
![Face Search Engine Demo](FaceSearchEngine_Demo.gif)

## Motivation
Oftentimes individuals are curious about their celebrity lookalike. Especially for young women who want to find their personal style, finding their celebrity lookalikes and following celebrity fashion icons’ dressing style will be a shortcut to dress fashionably.

For this reason, the researchers sought to create a face search engine. When the user uploads a photo with multiple faces as the input, all the faces in that photo can be detected and for each of the detected faces, a number of K similar faces from the celebrity dataset as well as their similarity scores will be shown in the output.

## Data Source 
A collection of over 1 million celebrity photos are provided by Ann Arbor Algorithms Inc. as the data source. In our GitHub demo, we used part of these photos and adding additional known faces for validation.

This project aims to construct a face search engine using the techologies of face detection, face feature extraction and information retrieval. We use [SeetaFaceEngine](https://github.com/seetaface/SeetaFaceEngine) and its python API [pyseeta](https://github.com/TuXiaokang/pyseeta) to detect and label the faces from a given image, and use [FaceNet](https://github.com/davidsandberg/facenet) to extract the features of each face. The search engine is built upon the [donkey framework](https://github.com/aaalgo/donkey), which receive an uploaded image file and return K most similar faces and their origin images from our dataset. 

## 0. Environment
### Operating system
```bash
Distributor ID:	Ubuntu
Description:	Ubuntu 18.04.1 LTS
Release:	18.04
Codename:	bionic
```
### pyseeta
Just follow the instruction of pyseeta installation: https://github.com/TuXiaokang/pyseeta
### FaceNet pre-trained model
The pre-trained model can be downloaded from the following link. The file name need to change and make sure the following file are included in this directory:
```bash
20180402-114759.pb
model-20180402-114759.index
model-20180402-114759.meta
model-20180402-114759.data-00000-of-00001
```
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|--------------|
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
## 1. Face collection
In order to construct the search engine, the very first thing is to collect the faces, and manage their information in a well-defined file structure. In the final version of our project, we have 1000000 celebrity images in our database (data not shown here for privacy consideration), and the input of these data is a lists structured as `[barcode, thumbnail_url, origin_url]`. Suppose the barcode is acd8b762, the corresponding directory structure should be:
```bash
|-- data
    `-- b7
        `-- 62
            `-- acd8b762
                |-- 0.png
                |-- 1.png
                |-- 2.png
                |-- info
                |-- features.npy
```
0.png, 1.png, 2.png are the three clipped faces for acd8b762. Info is a binary file that store the face information in a dictionary, and features.npy stores the feature vectors for each clipped face. 
```python
{
    'key': barcode
    'url': origin_url
    'info': {
      'ID': i,                          # face id
      'left': face.left,                # bounding box coordinates
      'right': face.right,
      'top': face.top,
      'bottom': face.bottom,
      'height': image_height,
      'width': image_width,
      'channel': image_channel,
      'score': face.score,              # face score
      'vector': face_feature            # face extracted feature
                                        # When dealing with the data, this will be set as None, because these data are stored in the .npy file                     
    }
}
```
`face_detect.py` is used to detect faces, and generate the clipped face files and info. `face_vectorize.py` is used to generate the features.npy.

## 2. Construct search engine
The donkey framework have already compiled in this repository. To construct a new database for the search engine, one needs to run `bash reset.sh` to reset the database, and open server `./server` before insert the data.  
In `donkey.xml` we define the address and port of the server. When changing them in the XML file, the corresponding code in `fawn.py` should also change.  

To insert data, run `python3 insertDB.py` when the server is open. And `searchDB.py` is a sample code that show the example of searching similar faces. The similarity is defined as L2 distance. Here is a running example which finds the 5 most similar face:
```bash
[{'key': 'data/s0/63/acecs063/0', 'meta': 'https://thumbnail/url/acecs063.jpg', 'details': '', 'score': 0.7011203765869141}, 
 {'key': 'data/30/74/acea3074/4', 'meta': 'https://thumbnail/url/acea3074.jpg', 'details': '', 'score': 0.7307677268981934}, 
 {'key': 'data/03/90/acdv0390/1', 'meta': 'https://thumbnail/url/acdv0390.jpg', 'details': '', 'score': 0.737593948841095}, 
 {'key': 'data/u2/19/acebu219/0', 'meta': 'https://thumbnail/url/acebu219.jpg', 'details': '', 'score': 0.7520275115966797}, 
 {'key': 'data/x8/53/acdhx853/0', 'meta': 'https://thumbnail/url/acdhx853.jpg', 'details': '', 'score': 0.7825058698654175}]
```

## Future Work

1. The face detection. As discussed above, an improved algorithm is needed to detect only human faces and thus, to improve the performance of our Face Search Engine. 

2. A bigger database. A bigger database would enable higher generalizability so that it can match the input face with generally more similar faces. 

3. Another type of dataset. The potential application of our Face Search Engine is not limited to searching celebrity lookalike. It can also be applied to other kinds of face searching if we change the type of database. For example, with the criminal database, the police could identify possible criminals from the real-time CCTV photo. 

4. Improvement of the user interface. Due to the time limitation, we didn’t implement the complicated user interface and thus, a better user interface with proper guidelines and prompts should be integrated into our current design.
             
## Acknowledgements

The authors thank Prof. Qiaozhu Mei and GSI Shiyan Yan for their insightful instructions on the project. The authors also thank Dr. Wei Dong from Ann Arbor Algorithm Inc. for providing the dataset with over one million celebrity photos and technique support.
