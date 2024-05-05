python -u run.py \
--exp_name='experiment name' \
--path_data='...' \
--path_exp_tree='...' \
--repo_root='...' \
--cur_obj='TEyeD' \
--train_data_percentage=1.0 \
--random_dataloader \
--aug_flag=1 \
--reduce_valid_samples=0 \
--workers=8 \
--remove_spikes=1 \
--epochs=80 \
--batches_per_ep=4000 \
--lr=8e-3 \
--wd=2e-2 \
--batch_size=1 \
--frames=4 \
--early_stop_metric=3D \
--early_stop=20 \
--model='res_50_3' \
--net_rend_head \
--net_simply_head_tanh=1 \
--temp_n_angles=100 \
--temp_n_radius=50 \
--loss_w_ellseg=0.0 \
--loss_rend_vectorized \
--loss_w_rend_gt_2_pred=0.15 \
--loss_w_rend_pred_2_gt=0.15 \
--loss_w_rend_pred_2_gt_edge=0.0 \
--loss_w_rend_diameter=0.0 \
--loss_w_supervise=1 \
--loss_w_supervise_gaze_vector_3D_L2=2.5 \
--loss_w_supervise_gaze_vector_3D_cos_sim=2.5 \
--loss_w_supervise_gaze_vector_UV=0.0 \
--loss_w_supervise_eyeball_center=0.15 \
--loss_w_supervise_pupil_center=0.0 \
--do_distributed=0 \
--local_rank=0 \
--use_GPU=1 \
--mode='one_vs_one' \
--dropout=0 \
--use_ada_instance_norm_mixup=0
