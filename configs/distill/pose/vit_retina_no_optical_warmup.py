_base_ = [
    '../../_base_/datasets/face/face_center.py'
]

teacher_ckpt = "/root/caixin/RawSense/nolens_face_align/logs/a_no_optical_face/vit_retina_test_wflw1_no_optical_warmup/epoch_850.pth"
# student_ckpt = "/root/caixin/RawSense/nolens_face_align/logs/a_no_optical_face/vit_retina_test_wflw5_no_optical_warmup_shift/best_NME_epoch_927.pth"
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
    load_weight_path="logs/distill/face/vit2optical_bg_updater_rotate/epoch_90.pth",
    requires_grad_psf=False,
    n_psf_mask=1)

no_optical = dict(
    type='SoftPsfConv',
    feature_size=2.76e-05,
    sensor='IMX250',
    input_shape=[3, 308, 257],
    scene2mask=0.4,
    mask2sensor=0.002,
    target_dim=[164, 128],
    requires_grad=True,
    down="resize",
    do_optical=False,
    noise_type=None,
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
total_epochs = 3000
log_config = dict(
    interval=5,
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

# model setting
student = dict(
    type='mmpose.TopDown',
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
    # load_weights_path=teacher_ckpt,
    train_cfg=dict(),
    test_cfg=dict(flip_test=True))
teacher = dict(
   type='mmpose.TopDown',
   backbone=dict(
        type='T2T_ViT_optical',
        optical=no_optical,
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
    load_weights_path=teacher_ckpt,
    train_cfg=dict(),
    test_cfg=dict(flip_test=True))

algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMSegArchitecture',
        model=student,
    ),
    with_student_loss=True,
    with_teacher_loss=False,
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        teacher_norm_eval=True,
        components=[
            dict(
                student_module='keypoint_head.fc',
                teacher_module='keypoint_head.fc',
                  losses=[
                    dict(
                        type='DistanceWiseRKD',
                        name='distance_wise_loss',
                        loss_weight=0.04,
                        with_l2_norm=True),
                    dict(
                        type='AngleWiseRKD',
                        name='angle_wise_loss',
                        loss_weight=0.04,
                        with_l2_norm=True),
                ])
        ]),
)



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
        scale_factor=0.2),
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
    dict(type='TopDownGetBboxCenterScale', padding=2),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=1),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=15,
        scale_factor=0.2),
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
     dict(type='TopDownGenerateTargetRegression'),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs']),
]
test_pipeline = val_pipeline

data_root = '/mnt/workspace/RawSense/data/retina_single_face_coco/annotations_single_point'
image_root = '/mnt/workspace/RawSense/data/widerface/WIDER_train'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='FaceCenterDataset',
        ann_file=f'{data_root}/face_landmarks_retina_train.json',
        img_prefix=f'{image_root}/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='FaceCenterDataset',
        ann_file=f'{data_root}/face_landmarks_retina_val.json',
        img_prefix=f'{image_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='FaceCenterDataset',
        ann_file=f'{data_root}/face_landmarks_retina_val.json',
        img_prefix=f'{image_root}/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
checkpoint_config = dict(interval=10)
custom_hooks = [
    dict(type='VisualConvHook'),
    dict(type='VisualAfterOpticalHook'),
]
# custom_hooks = [
#     dict(type='VisualConvHook'),
#     dict(type='VisualAfterOpticalHook')
# ]
# evaluation = dict(interval=1, metric='accuracy')
# custom_hooks = [
#     dict(type='VisualConvHook'),
#     dict(type='VisualAfterOpticalHook')
# ]
# runner = dict(type='IterBasedRunner', max_iters=200000)
# checkpoint_config = dict(interval=1000)
evaluation = dict(interval=10)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
