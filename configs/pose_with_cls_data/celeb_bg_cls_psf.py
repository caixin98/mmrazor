
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

optical = dict(
    type='SoftPsfConv',
    feature_size=2.76e-05,
    sensor='IMX250',
    input_shape=[3, 308, 257],
    scene2mask=0.4,
    mask2sensor=0.002,
    target_dim=[164, 128],
    requires_grad=True,
    down="resize",
    noise_type="gaussian",
    load_weight_path="/root/caixin/RawSense/mmrazor/logs/distill/face/vit2optical_bg_updater_rotate/epoch_90.pth",
    requires_grad_psf=False,
    n_psf_mask=1)


optimizer = dict(type='AdamW',lr=5e-4, weight_decay=0.05)
lr_config = dict(
    policy='CosineAnnealingCooldown',
    min_lr=1e-5,
    cool_down_time=10,
    cool_down_ratio=0.1,
    by_epoch=True,
    warmup_by_epoch=True,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-6)

total_epochs = 1000
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=1,
    dataset_joints=1,
    dataset_channel=[
        list(range(1)),
    ],
    inference_channel=list(range(1)))

# model settings
model = dict(
    type='TopDown',
   backbone=dict(
        type='T2T_ViT_optical',
        optical=optical,
        image_size=168),
    # backbone=dict(
    #     type='T2T_ViT',
    #     img_size=64),
    neck=dict(type='GlobalAveragePooling'),
    keypoint_head=dict(
        type='DeepposeRegressionHead',
        in_channels=384,
        num_joints=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='SmoothL1Loss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(flip_test=False))

data_cfg = dict(
    image_size=[128, 128],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=2),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=1),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=15,
        scale_factor=0.3),
    dict(type='TopDownAffine'),
    dict(
        type='Propagated',
        mask2sensor=0.002,
        scene2mask=0.4,
        object_height=None,
        sensor='IMX250',
        single_psf=False,
        grayscale=False,
        input_dim=[128, 128, 3],
        output_dim=[308, 257, 3]),
    dict(type='AddBackground', img_dir='/mnt/workspace/RawSense/data/BG-20k/train',size = (100, 100)),
    # dict(
    #     type='NormalizeTensor',
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetRegression'),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs'
        ]),
]

val_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='Propagated',
                keys=['img'],
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.27,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[112, 96, 3],
                output_dim=[308, 257, 3]),
            dict(
                    type='TorchAffineRTS',
                    angle=(0, 30),
                    scale_factor=0.2,
                    # translate=(0.2, 0.2),
                    prob=1.0,
                ),
            dict(type='Affine2label',),
            # dict(type='AddBackground', img_dir='/mnt/workspace/RawSense/data/BG-20k/testval',size = (100, 100),is_tensor=True),
     
            # dict(type='Collect', keys=['img', 'affine_matrix'],meta_keys=['image_file','affine_matrix'])
            dict(type='Collect', keys=['img', 'affine_matrix','target','target_weight'],meta_keys=['image_file','affine_matrix'])
]
test_pipeline = val_pipeline


data = dict(
    samples_per_gpu=200,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=128),
    test_dataloader=dict(samples_per_gpu=128),
    train=dict(
        type='Celeb',
        img_prefix='/mnt/workspace/RawSense/data/celebrity/',
        imglist_root=
        '/mnt/workspace/RawSense/data/celebrity/celebrity_data.txt',
        label_root='/mnt/workspace/RawSense/data/celebrity/celebrity_label.txt',
        pipeline=train_pipeline,
      ),
    val=dict(
        type='LFW',
            load_pair = False,
            img_prefix='/mnt/workspace/RawSense/data/lfw/lfw-112X96',
            pair_file='/mnt/workspace/RawSense/data/lfw/pairs.txt',
        pipeline=val_pipeline),
    test=dict(
        type='LFW',
            load_pair = False,
            img_prefix='/mnt/workspace/RawSense/data/lfw/lfw-112X96',
            pair_file='/mnt/workspace/RawSense/data/lfw/pairs.txt',
            pipeline=test_pipeline),
)
checkpoint_config = dict(interval=1)
custom_hooks = [
    dict(type='VisualConvHook'),
    dict(type='VisualAfterOpticalHook')
]
# evaluation = dict(interval=1, metric='accuracy')
# custom_hooks = [
#     dict(type='VisualConvHook'),
#     dict(type='VisualAfterOpticalHook')
# ]
# runner = dict(type='IterBasedRunner', max_iters=200000)
# checkpoint_config = dict(interval=1000)
evaluation = dict(metric='NME')
# workflow=[('val', 1)]
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))