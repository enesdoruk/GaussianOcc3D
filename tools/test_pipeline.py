from mmdet.datasets import DATASETS
from mmseg.models import builder
from mmengine.registry import MODELS

import mmdet3d
from mmdet3d.models import builder as builder3d
from projects.occ_plugin.utils.spconv_voxelize import SPConvVoxelization
import torch
from torch.nn import functional as F
from mmdet.models import LOSSES
import numpy as np

import importlib
importlib.import_module('projects.occ_plugin.datasets.nuscenes_occ_dataset')
importlib.import_module('projects.occ_plugin.occupancy.lifter.gaussian_lifter_v2')
importlib.import_module('projects.occ_plugin.occupancy.backbones.gaussian_encoder.anchor_encoder_module')
importlib.import_module('projects.occ_plugin.loss.multi_loss')


data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    # 'input_size': (128, 256),
    # 'input_size': (256, 704),
    'input_size': (864, 1600),
    'src_size': (900, 1600),
    # image-view augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}


depth_gt_path = './data/depth_gt'
img_norm_cfg = None


bda_aug_conf = dict(
            # rot_lim=(-22.5, 22.5),
            rot_lim=(-0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

occ_path = "./data/nuScenes-Occupancy"
occ_size = [500, 500, 40]
empty_idx = 0  # noise 0-->255
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
visible_mask = False
socc_path = "/home/edoruk/dataset/nuscenes_dataset/nuscenes_occ/samples"


train_pipeline = [
    dict(type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True, data_config=data_config,
                sequential=False, aligned=True, trans_only=False, depth_gt_path=depth_gt_path,
                mmlabnorm=True, load_depth=True, img_norm_cfg=img_norm_cfg),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        input_modality=input_modality),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, occ_path=occ_path, grid_size=occ_size, use_vel=False,
            unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type="LoadOccupancySurroundOcc", occ_path=socc_path, semantic=True, use_ego=False),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points', 'occ_xyz', 'occ_label', 'occ_cam_mask']),
    
]

dataset_type = 'NuscOCCDataset'
data_root = 'data/nuscenes/'
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"

train_config=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_path,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        box_type_3d='LiDAR')


dataset = DATASETS.build(train_config)

sample = dataset.__getitem__(0)

img = sample['img_inputs'][0].cuda()
meta = dataset.get_data_info(0)

img_backbone_cfg=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        with_cp=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch')

img_backbone = builder.build_backbone(img_backbone_cfg).cuda()
out_img_backbone = img_backbone(img)

embed_dims = 128
num_levels = 4

img_neck_config=dict(
         type="FPN",
         num_outs=num_levels,
         start_level=1,
         out_channels=embed_dims,
         add_extra_convs="on_output",
         relu_before_extra_convs=True,
         in_channels=[256, 512, 1024, 2048],
     )
     

img_neck = builder3d.build_neck(img_neck_config).cuda()
out_img_neck = img_neck(out_img_backbone)


semantics = True
semantic_dim = 17
phi_activation = 'loop'
include_opa = True
embed_dims = 128

lifter_cfg=dict(
        type='GaussianLifterV2',
        num_anchor=19200,
        embed_dims=embed_dims,
        anchor_grad=False,
        feat_grad=False,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        num_samples=128,
        anchors_per_pixel=1,
        random_sampling=False,
        projection_in=None,
        initializer=dict(
            type="ResNetSecondFPN",
            img_backbone_out_indices=[0, 1, 2, 3],
            img_backbone_config=dict(
                type='ResNet',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN2d', requires_grad=False),
                norm_eval=True,
                style='caffe',
                with_cp=True,
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
                stage_with_dcn=(False, False, True, True)),
            neck_confifg=dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=[embed_dims] * 4,
                upsample_strides=[0.5, 1, 2, 4])),
        initializer_img_downsample=None,
        deterministic=False,
        random_samples=6400)

import pdb; pdb.set_trace()
lifter = MODELS.build(lifter_cfg)

pts_voxel_layer_cfg=dict(
        num_point_features=5,
        max_num_points=10, 
        point_cloud_range=point_cloud_range,
        voxel_size=[0.1, 0.1, 0.1],  # xy size follow centerpoint
        max_voxels=(90000, 120000))

pts_voxel_layer = SPConvVoxelization(**pts_voxel_layer_cfg).cuda()

voxels, coors, num_points = [], [], []
for res in [sample['points']._data.cuda()]:
    res_voxels, res_coors, res_num_points = pts_voxel_layer(res)
    voxels.append(res_voxels)
    coors.append(res_coors)
    num_points.append(res_num_points)
voxels = torch.cat(voxels, dim=0)
num_points = torch.cat(num_points, dim=0)
coors_batch = []
for i, coor in enumerate(coors):
    coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
    coors_batch.append(coor_pad)
coors_batch = torch.cat(coors_batch, dim=0)

pts_voxel_encoder_cfg =dict(type='HardSimpleVFE', num_features=5)
pts_voxel_encoder = mmdet3d.models.build_voxel_encoder(pts_voxel_encoder_cfg)

out_pts_voxel_encoder = pts_voxel_encoder(voxels, num_points, coors_batch)

numC_Trans = 80
pts_middle_encoder_cfg=dict(
        type='SparseEncoderHD',
        in_channels=5,
        sparse_shape=[81, 1024, 1024],
        output_channels=numC_Trans,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        encoder_strides=(2, 2, 2, 1),
        block_type='basicblock',
        fp16_enabled=False,
    )

pts_middle_encoder = builder3d.build_neck(pts_middle_encoder_cfg).cuda()

xyz = coors_batch[:, [3, 2, 1]].unsqueeze(0).float().contiguous()
batch_size = coors_batch[-1, 0] + 1

out_pts_middle_encoder = pts_middle_encoder(out_pts_voxel_encoder, coors_batch, pts=xyz, batch_size=batch_size)
out_pts_middle_encoder = out_pts_middle_encoder['pts_feats'][0].squeeze(0).permute(1,0,2,3)


out_pts_lifter = lifter([out_pts_middle_encoder.unsqueeze(0)])

feature_maps =  [x.unsqueeze(0) for x in out_img_neck]
out_img_lifter = lifter(feature_maps)


num_groups = 4
use_deformable_func = True  # setup.py needs to be executed
scale_range = [0.1, 0.6]
xyz_coordinate = 'polar'

anchor = out_img_lifter['representation'].cuda()
instance_feature = out_img_lifter['rep_features'].cuda()

meta['focal_positions'] = torch.tensor(meta['focal_positions']).unsqueeze(0).cuda()
meta['cam_positions'] = torch.tensor(meta['cam_positions']).unsqueeze(0).cuda()
meta['projection_mat'] = torch.tensor(meta['projection_mat']).unsqueeze(0).cuda()
meta['occ_xyz'] = torch.tensor(dataset[0]['occ_xyz']).unsqueeze(0).cuda()
meta['occ_label'] = torch.tensor(dataset[0]['occ_label']).unsqueeze(0).cuda()
meta['occ_cam_mask'] = torch.tensor(dataset[0]['occ_cam_mask']).unsqueeze(0).cuda()


num_decoder = 4
num_single_frame_decoder = 1

encoder_cfg=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims, 
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
        ),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="add",
            kps_generator=dict(
                type="SparseGaussian3DKeyPointsGenerator",
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
                pc_range=point_cloud_range,
                scale_range=scale_range
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=point_cloud_range,
            scale_range=scale_range,
            restrict_xyz=True,
            unit_xyz=[4.0, 4.0, 1.0],
            refine_manual=[0, 1, 2],
            phi_activation=phi_activation,
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='softplus',
        ),
        spconv_layer=dict(
            _delete_=True,
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=point_cloud_range,
            grid_size=[0.5, 0.5, 0.5],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
        ),
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * num_single_frame_decoder + [
            "spconv",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * (num_decoder - num_single_frame_decoder),
    )


encoders = MODELS.build(encoder_cfg).cuda()

import pdb; pdb.set_trace()
out_encoders = encoders(out_img_lifter['anchor_init'].unsqueeze(0).cuda(), instance_feature.cuda(), feature_maps, meta)

head_cfg=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim + 1,
        empty_args=dict(
            _delete_=True,
            mean=[0, 0, -1.0],
            scale=[100, 100, 8.0],
        ),
        with_empty=True,
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=200, W=200, D=16,
            pc_min=[-50.0, -50.0, -5.0],
            grid_size=0.5),
    )

head = MODELS.build(head_cfg).cuda()


out_head = head(out_encoders['representation'], meta)


loss_cfg = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='OccupancyLoss',
            weight=1.0,
            empty_label=17,
            num_classes=18,
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0,
                loss_voxel_lovasz_weight=1.0),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=17,
            manual_class_weight=[
                1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ])
        ])



loss = LOSSES.build(loss_cfg)
out_loss = loss(out_head)


for idx, pred in enumerate(out_head['final_occ']):
    class_indices = list(range(1, 17))
    num_classes = len(class_indices)
    empty_label=17
    label_str=['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation'],
    use_mask=True
    dataset_empty_label=17
    filter_minmax=False

    outputs = pred
    targets = out_head['sampled_label'][idx]
    mask = out_head['occ_mask'][idx].flatten()

    total_seen = torch.zeros(num_classes+1).cuda()
    total_correct = torch.zeros(num_classes+1).cuda()
    total_positive = torch.zeros(num_classes+1).cuda()

    if not isinstance(targets, (torch.Tensor, np.ndarray)):
        assert mask is None
        labels = torch.from_numpy(targets['semantics']).cuda()
        masks = torch.from_numpy(targets['mask_camera']).bool().cuda()
        targets = labels
        targets[targets == dataset_empty_label] = empty_label
        if filter_minmax:
            max_z = (targets != empty_label).nonzero()[:, 2].max()
            min_z = (targets != empty_label).nonzero()[:, 2].min()
            outputs[..., (max_z + 1):] = empty_label
            outputs[..., :min_z] = empty_label
        if use_mask:
            outputs = outputs[masks]
            targets = targets[masks]
    else:
        if mask is not None:
            outputs = outputs[mask]
            targets = targets[mask]

    for i, c in enumerate(class_indices):
        total_seen[i] += torch.sum(targets == c).item()
        total_correct[i] += torch.sum((targets == c)
                                            & (outputs == c)).item()
        total_positive[i] += torch.sum(outputs == c).item()

        total_seen[-1] += torch.sum(targets != empty_label).item()
        total_correct[-1] += torch.sum((targets != empty_label)
                                        & (outputs != empty_label)).item()
        total_positive[-1] += torch.sum(outputs != empty_label).item()
    
    occ_iou = total_correct[-1] / (total_seen[-1]
                                            + total_positive[-1]
                                            - total_correct[-1])
