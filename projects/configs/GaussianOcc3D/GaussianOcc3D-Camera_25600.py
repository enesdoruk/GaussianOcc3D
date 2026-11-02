_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

plugin = True
plugin_dir = "projects/occ_plugin/"
img_norm_cfg = None
occ_path = "./data/nuScenes-Occupancy"
depth_gt_path = './data/depth_gt'
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
socc_path = "/home/edoruk/dataset/nuscenes_dataset/nuscenes_occ/samples"

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'

point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
occ_size = [500, 500, 40]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]  # 0.4
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_channels = [80, 160, 320, 640]
empty_idx = 0  # noise 0-->255
visible_mask = False

dataset_type = 'NuscOCCDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

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


numC_Trans = 128
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)
embed_dims = 128
num_levels = 4
semantic_dim = 17

semantics = True
phi_activation = 'sigmoid'
include_opa = True
num_anchor = 25600
xyz_coordinate = 'cartesian'
scale_range = [0.08, 0.64]
num_decoder = 4
num_single_frame_decoder = 1
num_groups = 4
use_deformable_func = True 

model = dict(
    type='GaussianOcc3D',
    version='RGB',
    loss_norm=False,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        with_cp=False,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), 
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type="FPN",
         num_outs=num_levels,
         start_level=1,
         out_channels=embed_dims,
         add_extra_convs="on_output",
         relu_before_extra_convs=True,
         in_channels=[256, 512, 1024, 2048],),
    gauss_lifter_img=dict(
        type='GaussianLifter',
        num_anchor=num_anchor,
        embed_dims=embed_dims,
        anchor_grad=True,
        feat_grad=False,
        phi_activation=phi_activation,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        ),
    empty_idx=empty_idx,
    gauss_encoder=dict(
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
            feedforward_channels=embed_dims * 2,
        ),
        deformable_model_img=dict(
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
                num_learnable_pts=2,
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
        version='RGB',
        operation_order=[
            "deformable_img",
            "ffn",
            "norm",] +
                
        [  "spconv",
            "norm",
            "deformable_img",
            "ffn",
            "norm",
            "refine",] +
        
        [  "spconv",
            "norm",
            "ffn",
            "norm",
            "refine",] +
        
        [  "spconv",
            "norm",
            "ffn",
            "norm",
            "refine",]
    ),
    head = dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim+1,
        empty_args=dict(
                        _delete_=True,
                        mean=[0, 0, -1.0],
                        scale=[100, 100, 8.0]),
        with_empty=True,
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=200, W=200, D=16,
            pc_min=[-50.0, -50.0, -5.0],
            grid_size=0.5),
        loss = dict(
            type='MultiLoss',
            loss_cfgs=[
                dict(
                    type='OccupancyLoss',
                    weight=1.0,
                    empty_label=17,
                    num_classes=semantic_dim+1,
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
                        1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ]
                    )
                ])
    )
)


bda_aug_conf = dict(
            # rot_lim=(-22.5, 22.5),
            rot_lim=(-0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5)

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
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, use_ego=False, occ_path=occ_path, socc_path=socc_path, grid_size=occ_size, use_vel=False,
            unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomOccCollect3D', keys=['img_inputs', 'gt_occ', 'points'],
          meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token', 'occ_xyz', 'occ_label', 'occ_cam_mask', 'projection_mat', 'focal_positions', 'cam_positions']),
]



test_pipeline = [
    dict(type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config, depth_gt_path=depth_gt_path,
         sequential=False, aligned=True, trans_only=False, mmlabnorm=True, img_norm_cfg=img_norm_cfg),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        input_modality=input_modality,
        is_train=False),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, occ_path=occ_path, socc_path=socc_path,grid_size=occ_size, use_vel=False,
        unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='CustomOccCollect3D', keys=['img_inputs', 'gt_occ', 'points'],
          meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token', 'occ_xyz', 'occ_label', 'occ_cam_mask', 'projection_mat', 'focal_positions', 'cam_positions']),
]


test_config=dict(
    type=dataset_type,
    occ_root=occ_path,
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range,
)

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

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=20)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC',
    rule='greater',
)

log_config = dict(
    interval=50,  
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='OCC3D',
                name='occ_mamba_baseline',
                config=dict(
                    dataset='Nuscenes',
                    model='occ_mamba_baseline'
                )
            ),
            log_artifact=True 
        )
    ])