# This code performs inference on a set of images using the Model aware 3 D Gaze Eye netowrk provided in https://github.com/dimitris-christodoulou57/Model-aware_3D_Eye_Gaze
# License and copyright according to that provided in https://github.com/dimitris-christodoulou57/Model-aware_3D_Eye_Gaze

'''

The MIT License

Copyright (c) 2023 Dimitrios Christodoulou, Nikola Popovic, Danda Pani Paudel, Xi Wang, Luc Van Gool

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, and Gabriel Diaz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys, os
import cv2
pathdir = os.path.join( os.path.dirname(__file__), 'Model_aware_3D_Eye_Gaze')
print('pathdir: ', pathdir)
sys.path.append(pathdir)

from Model_aware_3D_Eye_Gaze.args_maker import make_args


from Model_aware_3D_Eye_Gaze.models_mux import model_dict
from Model_aware_3D_Eye_Gaze.helperfunctions.utils import move_to_single, FRN_TLU, do_nothing
from set_arguments import set_args


args = vars(make_args())

args = set_args(args)

args['alpha'] = 0.5
args['beta'] = 0.5
args['frames'] = 4

path_pretrained = 'last.pt'

norm = nn.BatchNorm2d
net = model_dict[args['model']](args, norm=norm, act_func=F.leaky_relu)

net_dict = torch.load(path_pretrained, map_location=torch.device('cpu'))

state_dict_single = move_to_single(net_dict['state_dict'])
net.load_state_dict(state_dict_single, strict=False)


#print('net_dict keys: ', net_dict.keys())

i = 0

datapath = 'data/trial 2/'
resultpath = 'data/result 2 trial 2/'


imgf = datapath + 'img' + str(i) + '.jpg'             # image path of whole face
img1 = datapath + 'just_eyes' + str(i) + '.jpg'       # image path of eyes extracted, a sequence of 4 frames
img2 = datapath + 'just_eyes' + str(i+1) + '.jpg'
img3 = datapath + 'just_eyes' + str(i+2) + '.jpg'
img4 = datapath + 'just_eyes' + str(i+3) + '.jpg'



'''
TEyeDpath = 'Model_aware_3D_Eye_Gaze/data/img'

img1 = TEyeDpath + str(i) + '.jpg'       # image path of eyes extracted, a sequence of 4 frames
img2 = TEyeDpath + str(i+1) + '.jpg'
img3 = TEyeDpath + str(i+2) + '.jpg'
img4 = TEyeDpath + str(i+3) + '.jpg'
imgf = img1
'''

im1o = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
im2o = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
im3o = cv2.imread(img3, cv2.IMREAD_GRAYSCALE)
im4o = cv2.imread(img4, cv2.IMREAD_GRAYSCALE)

imgface = cv2.imread(imgf)

# resizing the image to 320, 240 

imgwresize = 320
imghresize = 240

im1 = cv2.resize(im1o, (imgwresize, imghresize))
im2 = cv2.resize(im2o, (imgwresize, imghresize))
im3 = cv2.resize(im3o, (imgwresize, imghresize))
im4 = cv2.resize(im4o, (imgwresize, imghresize))

imgs = np.stack((im1, im2, im3, im4))
imgs = imgs.reshape((1,imgs.shape[0], imgs.shape[1], imgs.shape[2]))
previmgs = np.copy(imgs)


imgw = 640
imgh = 480

#frames = 1000

framestart = 0
frames = 3351


# for storing the results

resultsdict = {}

gaze_vector_list = []
x_results = np.zeros((0,1))
y_results = np.zeros((0,1))
imgx_results = np.zeros((0,1))
imgy_results = np.zeros((0,1))

for i in range(framestart, framestart + frames-3):
    
    
    #print('imgs: ', imgs)

    t = torch.from_numpy(imgs)

    #print('t shape: ', t.shape)

    #print('type(t): ', type(t))

    data_dict = {'image': t}
    out_dict = 0

    # output of network

    with torch.no_grad():
        out_dict, out_dict_valid = net(data_dict, args)


    #print('out_dict gaze_vector_3D: ', out_dict['gaze_vector_3D'])
   
    # below: -x/z and -y/z, out_dict['gaze_vector_3D'][0,0,0] is the z-component, -out_dict['gaze_vector_3D'][0,0,1] is tbe x-component and
    # -out_dict['gaze_vector_3D'][0,0,2] is the y-component

    x = out_dict['gaze_vector_3D'][0,0,1]/out_dict['gaze_vector_3D'][0,0,0]
    y = out_dict['gaze_vector_3D'][0,0,2]/out_dict['gaze_vector_3D'][0,0,0]

    # taking the x and y components of 3d gae vector, seems to be -out_dict['gaze_vector_3D'][0,0,1] for x and -out_dict['gaze_vector_3D'][0,0,2] for y

    #x = -out_dict['gaze_vector_3D'][0,0,1]
    #y = -out_dict['gaze_vector_3D'][0,0,2]

    gaze_vector_list.append(out_dict['gaze_vector_3D'])
    x_results = np.append(x_results, x)
    y_results = np.append(y_results, y)

    
    # projecting the points on a 2 D image of size imgw, imgh, assuming that the maximum value of x/z and y/z can be +1.5 or -1.5
    # uncomment this when using x/z and y/z

    imgx = int((x/3)*imgw + imgw/2)
    imgy = int((y/3)*imgh + imgh/2)

    
    # projecting the points on a 2 D image of size imgw, imgh
    # uncomment this when uing the x and y components of the 3 D vector, after multiplying them by -1, x and y shall between -1 and 1 

    #imgx = max(0, min(int((x)*imgw/2 + imgw/2), imgw - 1))
    #imgy = max(0, min(int((y)*imgh/2 + imgh/2), imgh - 1))

    imgx_results = np.append(imgx_results, imgx)
    imgy_results = np.append(imgy_results, imgy)

    print('x: ', x)
    print('y: ', y, '\n')
    print('imgx: ', imgx)
    print('imgy: ', imgy)


    circleimg = 255*np.ones((imgh, imgw, 3))

    cv2.circle(circleimg, (imgx, imgy), 20, (0,185,255), -1)


    resultname = resultpath + 'img' + str(i) + '.jpg'     
    #resultname = 'data/result TEyeD/img' + str(i) + '.jpg'

    imgface = imgface[0:-120,:,:]   

    imgface = cv2.resize(imgface, (150,60))

    #imgface = cv2.resize(imgface, (160,120))   # for TEyeD Dataset

    # overlaying image of face on the circle image

    circleimg[-60:,-150:,:] = imgface

    #circleimg[-120:,-160:,:] = imgface   # for TEyeD Dataset

    cv2.imwrite(resultname, circleimg)

    print('i: ', i)

    if i < framestart + frames - 4:

        imgname4 = datapath + 'just_eyes' + str(i+4) + '.jpg'

        #imgname4 = TEyeDpath + str(i+4) + '.jpg'

        imgnew = cv2.imread(imgname4, cv2.IMREAD_GRAYSCALE)

        imgnew = cv2.resize(imgnew, (imgwresize, imghresize))

        previmgs = np.copy(imgs)

        imgs[0,0:3,:,:] = previmgs[0,1:4,:,:]

        imgs[0,3,:,:] = imgnew

        imgf = datapath + 'img' + str(i+1) + '.jpg'

        #imgf = TEyeDpath + str(i+1) + '.jpg'

        imgface = cv2.imread(imgf)

resultsdict = {'gaze_vector_list': gaze_vector_list, 'x_results': x_results, 'y_results': y_results, 'imgx_results': imgx_results, 'imgy_results': imgy_results}

with open(resultpath + 'results.npy', 'wb') as f:
    np.save(f, resultsdict, allow_pickle=True)