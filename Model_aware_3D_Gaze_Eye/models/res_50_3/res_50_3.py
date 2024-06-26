#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This started as a copy of https://bitbucket.org/RSKothari/multiset_gaze/src/master/ 
with additional changes and modifications to adjust it to our implementation. 

Copyright (c) 2021 Rakshit Kothari, Aayush Chaudhary, Reynold Bailey, Jeff Pelz, 
and Gabriel Diaz
"""
import copy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import sys
sys.path.append('..')

from models.resnet_encoder import resnet50
from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from models.dense_decoder import DenseNet_decoder
from models.regresion_module import regressionModuleEllseg
from models.transformer import Transformer

from models.utils import get_com, getSizes
from models.utils import construct_mask_from_ellipse
from models.utils import fix_ellipse_axis_angle

from helperfunctions.utils import detach_cpu_np, get_nparams
from helperfunctions.helperfunctions import plot_images_with_annotations
from helperfunctions.helperfunctions import my_ellipse

from extern.pytorch_revgrad.src.module import RevGrad


class res_50_3(nn.Module):
    def __init__(self,
                 args,
                 act_func=F.leaky_relu,
                 norm=nn.BatchNorm2d):
        super(res_50_3, self).__init__()

        self.N_win = args['frames']
        self.net_ellseg_head = args['net_ellseg_head']
        self.net_rend_head = args['net_rend_head']
        self.net_simply_head = args['net_simply_head']

        assert (self.net_rend_head or self.net_simply_head)
        assert not (self.net_rend_head and self.net_simply_head)

        self.alpha = 1.0

        self.extra_depth = args['extra_depth']

        self.sizes = getSizes(args['base_channel_size'],
                              args['growth_rate'],
                              blks=args['num_blocks'])

        self.equi_var = args['equi_var']

        # self.enc = DenseNet_encoder(args,
        #                             in_c=1,
        #                             act_func=act_func, norm=norm)

        self.enc = resnet50(pretrained=args['pretrained_resnet'], **{'in_chans': 1}) # TODO enable parametrizing pretrained weights
        assert not self.net_ellseg_head # TODO Enable using the ellseg head (adapt feature pyramid size)

        self.global_pool = SelectAdaptivePool2d(
            pool_type='avg',
            flatten=True,
            input_fmt='NCHW',
        )

        if self.net_ellseg_head:
            # Fix the decoder norm to IN unless adopting AdaIN. The original paper
            # of AdaIN recommends disabling normalization for the decoder.
            if args['use_ada_instance_norm'] or args['use_ada_instance_norm_mixup']:
                # Adaptive Instance norm suggests not using any normalization in
                # the decoder when paired with adaptive instance normalization. It
                # is a design choice which improves performance apparantly.
                self.dec = DenseNet_decoder(args,
                                            out_c=3,
                                            act_func=act_func,
                                            norm='none')
            else:
                self.dec = DenseNet_decoder(args,
                                            out_c=3,
                                            act_func=act_func,
                                            norm=nn.InstanceNorm2d)

            if args['maxpool_in_regress_mod'] > 0:
                regress_pool = nn.MaxPool2d
            elif args['maxpool_in_regress_mod'] == 0:
                regress_pool = nn.AvgPool2d
            else:
                regress_pool = False
        
            # Fix the Regression module to InstanceNorm
            self.elReg = regressionModuleEllseg(self.sizes,
                                                sc=args['regress_channel_grow'],
                                                norm=nn.InstanceNorm2d,
                                                pool=regress_pool,
                                                dilate=args['dilation_in_regress_mod'],
                                                act_func=act_func,
                                                track_running_stats=args['track_running_stats'])

        # TODO: What is this? Does this also go inside args['net_ellseg_head']?
        self.make_aleatoric = args['make_aleatoric']
        if args['grad_rev']:
            self.grad_rev = True
            self.setDatasetInfo(args['num_sets'])
        else:
            self.grad_rev = False        
            
        if self.net_rend_head or self.net_simply_head:
            embed_dim=256
            self.rend_head_1 = nn.Linear(in_features=2048, out_features=embed_dim, bias=True)
            self.eye_3d = Transformer(num_tokens=self.N_win, embed_dim=embed_dim,
                                      depth=3, num_heads=8, mlp_ratio=2.,
                                      qkv_bias=True, qk_norm=False, 
                                      init_values=None, pre_norm=False, fc_norm=True,
                                      drop_rate=0., pos_drop_rate=0., proj_drop_rate=0., 
                                      attn_drop_rate=0., drop_path_rate=0.,
                                      weight_init='', norm_layer=None, act_layer=None)
            if self.net_simply_head:
                self.n_feat_eye_diff = 5
                self.n_feat_eye_same = 2
            elif self.net_rend_head:
                self.n_feat_eye_diff = 7
                self.n_feat_eye_same = 7
            else:
                raise NotImplementedError

            self.rend_out_diff = nn.Linear(in_features=embed_dim, 
                                           out_features=self.n_feat_eye_diff, bias=True)
            self.rend_out_same = nn.Linear(in_features=embed_dim, 
                                           out_features=self.n_feat_eye_same, bias=True)


    def setDatasetInfo(self, numSets=2):
        # Produces a 1x1 conv2D layer which directly maps bottleneck to the
        # appropriate dataset ID. This mapping is done for each pixel, ensuring

        inChannels = self.sizes['enc']['op'][-1]

        self.numSets = numSets
        self.grad_rev_layer = RevGrad()
        self.dsIdentify_lin = nn.Conv2d(in_channels=inChannels,
                                        out_channels=numSets,
                                        kernel_size=1,
                                        bias=True)

    def forward(self, data_dict, args):
        
        # Cast to float32, move to device and add a dummy color channel
        if 'device' not in self.__dict__:
            self.device = next(self.parameters()).device

        out_dict_valid = True

        # Move image data to GPU
        x = data_dict['image'].to(torch.float32).to(self.device,
                                                    non_blocking=True)
        
        B, N, H, W = x.shape
        assert N == self.N_win
        # Merge batch and window size into one dimension
        x = rearrange(x, 'B N H W -> (B N) 1 H W')
        start = time.time()
        
        # Generate latent space bottleneck representation, as well as a featuremulti-scale pyramid
        latent, enc_op = self.enc(x) 
        #latent = enc_op[-1].flatten(start_dim=-2).mean(dim=-1)
        latent =  self.global_pool(latent)
        
        out_dict = {}
        out_dict['dT'] = 0 # FIXME quich and stupid fix
        out_dict['latent'] = latent.detach().cpu()
        if self.net_ellseg_head:

            elOut, elConf = self.elReg(enc_op[-1])

            op = self.dec(enc_op[:-1], enc_op[-1])

            end = time.time()

            if torch.any(torch.isnan(op)) or torch.any(torch.isinf(op)):

                #print(data_dict['archName'])
                #print(data_dict['im_num'])
                #print('WARNING! Convergence failed!')
                #print('Network predicted NaNs or Infs')
                out_dict_valid = False

            #     from scripts import detach_cpu_numpy

            #     plot_images_with_annotations(detach_cpu_numpy(data_dict),
            #                                 args,
            #                                 write='./FAILURE.jpg',
            #                                 remove_saturated=False,
            #                                 is_list_of_entries=False,
            #                                 is_predict=False,
            #                                 show=False)

            #     import sys
            #     sys.exit('Network predicted NaNs or Infs')

            # %% Gradient reversal on latent space
            # Observation: Adaptive Instance Norm removes domain info quite nicely
            if self.grad_rev:
                self.grad_rev_layer._alpha = torch.tensor(self.alpha,
                                                        requires_grad=False,
                                                        device=self.device)
                # Gradient reversal to remove DS bias
                ds_predict = self.dsIdentify_lin(self.grad_rev_layer(enc_op[-1]))
            else:
                ds_predict = []

            # %% Choose EllSeg proposed ellipse measures

            # Get center of ellipses from COM
            pred_pup_c = get_com(op[:, 2, ...], temperature=4)
            pred_iri_c = get_com(-op[:, 0, ...], temperature=4)

            # Un-merge batch and window size
            elOut = rearrange(elOut, '(B N) d -> B N d', B=B, N=N)
            elConf = rearrange(elConf, '(B N) d -> B N d', B=B, N=N)
            op = rearrange(op, '(B N) C H W -> B N C H W', B=B, N=N)
            pred_pup_c = rearrange(pred_pup_c, '(B N) d -> B N d', B=B, N=N)
            pred_iri_c = rearrange(pred_iri_c, '(B N) d -> B N d', B=B, N=N)

            # Ensure batch doesn't get squeezed if == 1
            pred_pup_c = pred_pup_c.reshape(B, N, -1)
            pred_iri_c = pred_iri_c.reshape(B, N, -1)

            if torch.any(torch.isnan(pred_pup_c)) or\
            torch.any(torch.isnan(pred_iri_c)):

                
                print('WARNING! Convergence failed!')
                print('Pupil or Iris centers predicted as NaNs')
                out_dict_valid = False

                #import sys
                #sys.exit('Pupil or Iris centers predicted as NaNs')

            # Append pupil and iris ellipse parameter predictions from latent space
            pupil_ellipse_norm = torch.cat([pred_pup_c, elOut[:, :, 7:10]], dim=-1)
            iris_ellipse_norm = torch.cat([pred_iri_c, elOut[:, :, 2: 5]], dim=-1)
            
            # %% Convert predicted ellipse back to proper scale
            if self.equi_var:
                sc = max([W, H])
                Xform_to_norm = np.array([[2/sc, 0, -1],
                                        [0, 2/sc, -1],
                                        [0, 0,     1]])
            else:
                Xform_to_norm = np.array([[2/W, 0, -1],
                                        [0, 2/H, -1],
                                        [0, 0,    1]])

            Xform_from_norm = np.linalg.inv(Xform_to_norm)

            out_dict['dT'] = end - start
            out_dict['iris_ellipse'] = np.zeros((B, N, 5))
            out_dict['pupil_ellipse'] = np.zeros((B, N, 5))

            

            for b in range(B):
                for n in range(N):
                    # Read each normalized ellipse in a loop and unnormalize it
                    try:
                        temp_var = detach_cpu_np(iris_ellipse_norm[b, n, ])
                        temp_var = fix_ellipse_axis_angle(temp_var)
                        temp_var = my_ellipse(temp_var).transform(Xform_from_norm)[0][:5]
                    except Exception:
                        print(temp_var)
                        print('Incorrect norm iris: {}'.format(temp_var.tolist()))
                        temp_var = np.ones(5, )

                    out_dict['iris_ellipse'][b, n, ...] = temp_var

                    try:
                        temp_var = detach_cpu_np(pupil_ellipse_norm[b, n, ])
                        temp_var = fix_ellipse_axis_angle(temp_var)
                        temp_var = my_ellipse(temp_var).transform(Xform_from_norm)[0][:5]
                    except Exception:
                        print(temp_var)
                        print('Incorrect norm pupil: {}'.format(temp_var.tolist()))
                        temp_var = np.ones(5, )

                    out_dict['pupil_ellipse'][b, n, ...] = temp_var

            # %% Pupil and Iris mask construction from predicted ellipses
            # TODO Change construct_mask_from_ellipse() to not require reshape
            # TODO DImitrios already wrote a B,N bersion in DenseElNet1
            out_dict['pupil_ellipse'] = rearrange(out_dict['pupil_ellipse'], 'B N d -> (B N) d')
            pupil_mask_recon = construct_mask_from_ellipse(out_dict['pupil_ellipse'], (H,W))
            out_dict['pupil_ellipse'] = rearrange(out_dict['pupil_ellipse'], '(B N) d -> B N d', B=B, N=N)
            out_dict['iris_ellipse'] = rearrange(out_dict['iris_ellipse'], 'B N d -> (B N) d')
            iris_mask_recon = construct_mask_from_ellipse(out_dict['iris_ellipse'], (H,W))
            out_dict['iris_ellipse'] = rearrange(out_dict['iris_ellipse'], '(B N) d -> B N d', B=B, N=N)

            pd_recon_mask = np.zeros(pupil_mask_recon.shape, dtype=int)
            pd_recon_mask[iris_mask_recon.astype(bool)] = 1
            pd_recon_mask[pupil_mask_recon.astype(bool)] = 2

            # %% Save out predicted data and return
            out_dict['mask'] = torch.argmax(op, dim=2).detach().cpu().numpy()
            out_dict['mask_recon'] = pd_recon_mask

            out_dict['pupil_ellipse_norm'] = pupil_ellipse_norm
            out_dict['iris_ellipse_norm'] = iris_ellipse_norm

            out_dict['pupil_ellipse_norm_regressed'] = elOut[:, :, 5:]
            out_dict['iris_ellipse_norm_regressed'] = elOut[:, :, :5]

            out_dict['pupil_center'] = out_dict['pupil_ellipse'][:, :, :2]
            out_dict['iris_center'] = out_dict['iris_ellipse'][:, :, :2]

            if self.make_aleatoric:
                out_dict['pupil_conf'] = elConf[:, :, 5:]
                out_dict['iris_conf'] = elConf[:, :, :5]
            else:
                out_dict['pupil_conf'] = torch.zeros_like(elConf)
                out_dict['iris_conf'] = torch.zeros_like(elConf)

            out_dict['ds_onehot'] = ds_predict
            out_dict['predict'] = op
            out_dict['latent'] = latent.detach().cpu()

        
        # Un-merge batch and window size
        # TODO Rearange x, latent, enc_op
        latent = rearrange(latent, '(B N) C-> B N C', B=B, N=N)

        if self.net_rend_head or self.net_simply_head:
            eye_3d_out = self.rend_head_1(latent)
            eye_3d_out = self.eye_3d(eye_3d_out)

            eye_3d_out_same = self.rend_out_same(eye_3d_out.mean(dim=1)).unsqueeze(1)
            if self.net_simply_head:
                eye_3d_out_same = eye_3d_out_same.expand(-1,N,-1)
                eye_3d_out_same = rearrange(eye_3d_out_same, 'B N d -> (B N) d')
            eye_3d_out_diff = self.rend_out_diff(rearrange(eye_3d_out, 'B N C -> (B N) C'))

            if self.net_rend_head:
                eye_3d_out_diff = rearrange(eye_3d_out_diff, '(B N) C -> B N C', B=B, N=N)
            
            if self.net_rend_head:
                #eye_3d_out_same = torch.nn.functional.hardtanh_(eye_3d_out_same)
                #eye_3d_out_diff = torch.nn.functional.hardtanh_(eye_3d_out_diff)
                eye_3d_out_same = torch.tanh(eye_3d_out_same)
                eye_3d_out_diff = torch.tanh(eye_3d_out_diff)

                out_dict['L'] = eye_3d_out_same[..., 0]
                out_dict['r_iris'] = eye_3d_out_same[..., 1]
                out_dict['T'] = eye_3d_out_same[..., 2:5]
                out_dict['focal'] = eye_3d_out_same[..., -2:]

                out_dict['R'] = eye_3d_out_diff[..., :3]
                out_dict['r_pupil'] = eye_3d_out_diff[..., 3]
                #normalize the 3D gaze vector
                norm_3D_gaze = eye_3d_out_diff[..., -3:] / (torch.norm(eye_3d_out_diff[..., -3:],dim=-1,keepdim=True) + 1e-9)
                out_dict['gaze_vector_3D'] = norm_3D_gaze
            elif self.net_simply_head:
                #eye_3d_out_same = torch.nn.functional.hardtanh_(eye_3d_out_same)
                #eye_3d_out_diff = torch.nn.functional.hardtanh_(eye_3d_out_diff)
                if args['net_simply_head_tanh']:
                    eye_3d_out_same = torch.tanh(eye_3d_out_same)
                    eye_3d_out_diff = torch.tanh(eye_3d_out_diff)
                else: 
                    eye_3d_out_same = eye_3d_out_same
                    eye_3d_out_diff = eye_3d_out_diff

                out_dict['eyeball_c_UV'] = eye_3d_out_same

                out_dict['pupil_c_UV'] = eye_3d_out_diff[..., :2]
                #normalize the 3D gaze vector
                norm_3D_gaze = eye_3d_out_diff[..., -3:] / (torch.norm(eye_3d_out_diff[..., -3:],dim=-1,keepdim=True) + 1e-9)
                out_dict['gaze_vector_3D'] = norm_3D_gaze
            else:
                raise NotImplementedError

        return out_dict, out_dict_valid


from functools import partial
from itertools import repeat
import collections.abc


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
    