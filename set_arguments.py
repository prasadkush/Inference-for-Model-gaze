'''

The MIT License

Copyright (c) 2023 Dimitrios Christodoulou, Nikola Popovic, Danda Pani Paudel, Xi Wang, Luc Van Gool

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, and Gabriel Diaz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''



import os
import torch
import random
import warnings
import numpy as np
#import wandb
import string

from datetime import datetime   



def set_args(args):
    args['batches_per_ep'] = 10
    args['epochs'] = 2
    #args['path_model'] = '/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Results/T_N0_H2_F4_C_fKysY_27_04_23_23_58_41/results/2.pt'
    args['frames'] = 4 
    args['batch_size']= 1
        #args['use_GPU']=1

        #args['model'] = 'DenseEl3'
    args['model'] = 'res_50_3'
    args['early_stop_metric'] = '3D'
    args['random_dataloader'] = False
    args['temp_n_angles'] = 72
    args['temp_n_radius'] = 8
    args['net_rend_head'] = True
    args['loss_w_rend_pred_2_gt_edge'] = 0.0
    args['loss_w_rend_gt_2_pred'] = 0.0
    args['loss_w_rend_pred_2_gt'] = 0.0
    args['loss_w_rend_diameter'] = 0.0
    args['net_ellseg_head'] = False
    args['loss_w_ellseg'] = 0.0
    args['loss_rend_vectorized'] = True
    args['detect_anomaly'] = 0

        #supervised_loss
    args['net_simply_head'] = False
    args['loss_w_supervise'] = 1
    args['loss_w_supervise_eyeball_center'] = 0.0
    args['loss_w_supervise_pupil_center'] = 0.0
    args['loss_w_supervise_gaze_vector_UV'] = 0.0
    args['loss_w_supervise_gaze_vector_3D_L2'] = 5.0
    args['loss_w_supervise_gaze_vector_3D_cos_sim'] = 0.0
    args['scale_bound_eye'] = 'version_1'
    args['pretrained_resnet'] = False
    args['net_simply_head_tanh'] = 0
    
    args['grad_clip_norm'] = 0.1
    args['optimizer_type'] = 'adamw_cos'

        #args['only_test'] = 1

        # args['only_test']=0
        # args['only_valid']=0
    args['train_data_percentage'] = 1.0
        #args['use_pkl_for_dataload'] = True
        #args['path_data']='/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Datasets/All'
        #args['path_exp_tree']='/srv/beegfs02/scratch/aegis_cvl/data/dchristodoul/Results/'
        #args['weights_path'] = '/home/nipopovic/MountedDirs/euler/work_specta/experiment_logs/dimitrios_gaze/TG_trfT_2e-4_perc_0.005_res_50_3_BF_4_4_Nang72_Nrad_8_vpsEX_21_06_23_10_39_38/results/last.pt'
        #args['path_model']='/home/nipopovic/MountedDirs/euler/work_specta/experiment_logs/dimitrios_gaze/TG_trfT_2e-4_perc_0.005_res_50_3_BF_4_4_Nang72_Nrad_8_vpsEX_21_06_23_10_39_38/results/last.pt'


    #path_dict['repo_root'] = args['repo_root']
    #path_dict['path_data'] = args['path_data']

    #change the number of frames to predifine 10
    #to load the pkl file with 10 images
    if args['use_pkl_for_dataload']:
        args['frames']=4

    # %% DDP essentials

    if args['do_distributed']:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        world_size = torch.distributed.get_world_size()

    else:
        world_size = 1

    global batch_size_global
    batch_size_global = int(args['batch_size']*world_size)
 
    #torch.cuda.set_device(args['local_rank'])
    args['world_size'] = world_size
    args['batch_size'] = int(args['batch_size']/world_size)

    # %%
    #torch.backends.cudnn.deterministic = False
    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = True

    # Set seeds
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    return args