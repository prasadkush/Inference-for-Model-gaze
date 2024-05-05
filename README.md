# Inference-for-Model-gaze

This is an inference code for performing inference on a set of 4 images as an input to the inference network in https://github.com/dimitris-christodoulou57/Model-aware_3D_Eye_Gaze

The weights used are the ones trained for Segmentation + Gaze + Center and provided in https://github.com/dimitris-christodoulou57/Model-aware_3D_Eye_Gaze

Instructions for installation:

The installation instructions for Model Aware 3D Eye Gaze package can be found in https://github.com/dimitris-christodoulou57/Model-aware_3D_Eye_Gaze


Instructions for setup:

The path of pretrained weights and data has to be provided in inference_images.py. If data is not already present, capture_face.py file provided can be used for saving face images and extracting the eyes from them.


Instructions for running:

python inference_images.py


