# Inference-for-Model-gaze

This is an inference code for performing inference on a set of 4 images as an input to the inference network in https://github.com/dimitris-christodoulou57/Model-aware_3D_Eye_Gaze

The weights used are the ones trained for Segmentation + Gaze + Center and provided in https://github.com/dimitris-christodoulou57/Model-aware_3D_Eye_Gaze

Instructions for installation:

The installation instructions for Model Aware 3D Eye Gaze package can be found in https://github.com/dimitris-christodoulou57/Model-aware_3D_Eye_Gaze

Note: The Model_aware_3D_Gaze_Eye folder contains code from https://github.com/dimitris-christodoulou57/Model-aware_3D_Eye_Gaze with very minor modifications in one or two files.

Instructions for setup:

The path of pretrained weights and data has to be provided in inference_images.py. If data is not already present, capture_face.py or capture_face_2.py file can be used for taking input from the web camera and saving face and eye images.


Instructions for running:

python inference_images.py


