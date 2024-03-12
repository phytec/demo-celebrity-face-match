AI Demo - Celebrity Face Match
================================================================================

Introduction
--------------------------------------------------------------------------------

This project is a demo application to show the capabilities of the embedded
neural processing unit (NPU) of the i.MX 8MPlus in combination with the PHYTEC
VM-016 camera.

The demo will recognize your face and extract embeddings via an artificial
neural network (ANN). Your embeddings will be compared with precalculated
embeddings from ~5k images of known Hollywood celebs. The ANN inference time
lasts around a mean of 0.016s. The whole process takes a bit longer (~1s).


HowTo
--------------------------------------------------------------------------------

First you need to download and extract the required tensorflow lite model,
previously created embeddings of the celebrities and their images.

```
wget https://download.phytec.de/Software/Linux/Applications/demo-celebrity-face-match-data-1.1.tar.gz
tar -xzf demo-celebrity-face-match-data-1.1.tar.gz
```

Second a PHYTEC VM-016 camera needs to be connected to the phyBOARD-Pollux
MIPI-CSI2 CSI1 connector.

Alternatively a USB camera can be used. This requires following change in
'aidemo.py'
```
 ...
 18 #CAMERA = 'VM-016'
 19 CAMERA = 'USB'
 ...
```

Now the demo can be started with:
```
python3 aidemo.py
```


Requirements
--------------------------------------------------------------------------------

 - python3

 Following python modules:
 - tflite_runtime	(see https://www.tensorflow.org/lite/guide/python)
 - opencv-contrib-python
 - PyGObject
 - pycairo

License
--------------------------------------------------------------------------------

This project is licensed under the *Apache License Version 2.0*. See the file
`LICENSE` for detailed information.
