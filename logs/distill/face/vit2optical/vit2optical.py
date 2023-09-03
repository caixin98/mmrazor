find_unused_parameters = True
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
dataset_type = 'Celeb'
num_classes = 93955
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0], to_rgb=True)
train_dir = '/mnt/workspace/RawSense/data/celebrity/'
train_imglist = '/mnt/workspace/RawSense/data/celebrity/celebrity_data.txt'
train_ann_file = '/mnt/workspace/RawSense/data/celebrity/celebrity_label.txt'
val_dir = '/mnt/workspace/RawSense/data/lfw/'
data = dict(
    workers_per_gpu=2,
    train=dict(
        type='Celeb',
        img_prefix='/mnt/workspace/RawSense/data/celebrity/',
        imglist_root=
        '/mnt/workspace/RawSense/data/celebrity/celebrity_data.txt',
        label_root='/mnt/workspace/RawSense/data/celebrity/celebrity_label.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(172, 172)),
            dict(type='Pad_celeb', size=(180, 172), padding=(0, 8, 0, 0)),
            dict(type='CenterCrop', crop_size=(112, 96)),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Propagated',
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.27,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[112, 96, 3],
                output_dim=[308, 257, 3]),
            dict(type='TorchAffineRTS', angle=(0, 30), prob=1.0),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label', 'affine_matrix'])
        ]),
    val=dict(
        type='LFW',
        img_prefix='/mnt/workspace/RawSense/data/lfw/lfw-112X96',
        pair_file='/mnt/workspace/RawSense/data/lfw/pairs.txt',
        pipeline=[
            dict(type='LoadImagePair'),
            dict(
                type='FlipPair',
                keys=['img1', 'img2'],
                keys_flip=['img1_flip', 'img2_flip']),
            dict(
                type='Propagated',
                keys=['img1', 'img1_flip', 'img2', 'img2_flip'],
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.27,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[112, 96, 3],
                output_dim=[308, 257, 3]),
            dict(type='TorchAffineRTS', angle=(0, 30), prob=1.0),
            dict(type='ToTensor', keys=['fold', 'label']),
            dict(
                type='StackImagePair',
                keys=['img1', 'img1_flip', 'img2', 'img2_flip'],
                out_key='img'),
            dict(
                type='Collect', keys=['img', 'fold', 'label', 'affine_matrix'])
        ]),
    test=dict(
        type='LFW',
        img_prefix='/mnt/workspace/RawSense/data/lfw/lfw-112X96',
        pair_file='/mnt/workspace/RawSense/data/lfw/pairs.txt',
        pipeline=[
            dict(type='LoadImagePair'),
            dict(
                type='FlipPair',
                keys=['img1', 'img2'],
                keys_flip=['img1_flip', 'img2_flip']),
            dict(
                type='Propagated',
                keys=['img1', 'img1_flip', 'img2', 'img2_flip'],
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.27,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[112, 96, 3],
                output_dim=[308, 257, 3]),
            dict(type='TorchAffineRTS', angle=(0, 30), prob=1.0),
            dict(type='ToTensor', keys=['fold', 'label']),
            dict(
                type='StackImagePair',
                keys=['img1', 'img1_flip', 'img2', 'img2_flip'],
                out_key='img'),
            dict(type='Collect', keys=['img', 'fold', 'label'])
        ]),
    train_dataloader=dict(samples_per_gpu=140),
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64))
custom_hooks = [
    dict(type='VisualConvHook', do_distall=True),
    dict(type='VisualAfterOpticalHook', do_distall=True)
]
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=20000,
    warmup_ratio=0.25)
checkpoint_config = dict(interval=10)
runner = dict(type='EpochBasedRunner', max_epochs=100)
evaluation = dict(interval=1, metric='accuracy')
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
teacher_ckpt = '/root/caixin/RawSense/nolens_mmcls/logs/a_no_optical_face/full_with_base/epoch_50.pth'
optical = dict(
    type='CropRotatePsfConv',
    feature_size=2.76e-05,
    sensor='IMX250',
    input_shape=[3, 306, 255],
    scene2mask=0.4,
    mask2sensor=0.002,
    target_dim=[164, 128],
    requires_grad=True,
    down='resize',
    noise_type='gaussian',
    angle=30,
    n_psf_mask=1)
no_optical = dict(
    type='SoftPsfConv',
    feature_size=2.76e-05,
    sensor='IMX250',
    input_shape=[3, 308, 257],
    scene2mask=0.4,
    mask2sensor=0.002,
    target_dim=[164, 128],
    do_optical=False,
    requires_grad=True,
    use_stn=False,
    down='resize',
    noise_type=None,
    n_psf_mask=1)
student = dict(
    type='mmcls.AffineFaceImageClassifier',
    backbone=dict(
        type='T2T_ViT_optical',
        optical=dict(
            type='CropRotatePsfConv',
            feature_size=2.76e-05,
            sensor='IMX250',
            input_shape=[3, 306, 255],
            scene2mask=0.4,
            mask2sensor=0.002,
            target_dim=[164, 128],
            requires_grad=True,
            down='resize',
            noise_type='gaussian',
            angle=30,
            n_psf_mask=1),
        image_size=168),
    neck=dict(
        type='GlobalDepthWiseNeck',
        in_channels=384,
        out_channels=128,
        kernel_size=(11, 11)),
    head=dict(
        type='IdentityClsHead',
        loss=dict(type='ArcMargin', out_features=93955)))
teacher = dict(
    type='mmcls.AffineFaceImageClassifier',
    backbone=dict(
        type='T2T_ViT_optical',
        optical=dict(
            type='SoftPsfConv',
            feature_size=2.76e-05,
            sensor='IMX250',
            input_shape=[3, 308, 257],
            scene2mask=0.4,
            mask2sensor=0.002,
            target_dim=[164, 128],
            do_optical=False,
            requires_grad=True,
            use_stn=False,
            down='resize',
            noise_type=None,
            n_psf_mask=1),
        apply_affine=True,
        image_size=168),
    neck=dict(
        type='GlobalDepthWiseNeck',
        in_channels=384,
        out_channels=128,
        kernel_size=(11, 11)),
    head=dict(
        type='IdentityClsHead',
        loss=dict(type='ArcMargin', out_features=93955)),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        '/root/caixin/RawSense/nolens_mmcls/logs/a_no_optical_face/full_with_base/epoch_50.pth'
    ))
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMClsArchitecture',
        model=dict(
            type='mmcls.AffineFaceImageClassifier',
            backbone=dict(
                type='T2T_ViT_optical',
                optical=dict(
                    type='CropRotatePsfConv',
                    feature_size=2.76e-05,
                    sensor='IMX250',
                    input_shape=[3, 306, 255],
                    scene2mask=0.4,
                    mask2sensor=0.002,
                    target_dim=[164, 128],
                    requires_grad=True,
                    down='resize',
                    noise_type='gaussian',
                    angle=30,
                    n_psf_mask=1),
                image_size=168),
            neck=dict(
                type='GlobalDepthWiseNeck',
                in_channels=384,
                out_channels=128,
                kernel_size=(11, 11)),
            head=dict(
                type='IdentityClsHead',
                loss=dict(type='ArcMargin', out_features=93955)))),
    with_student_loss=True,
    with_teacher_loss=False,
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=dict(
            type='mmcls.AffineFaceImageClassifier',
            backbone=dict(
                type='T2T_ViT_optical',
                optical=dict(
                    type='SoftPsfConv',
                    feature_size=2.76e-05,
                    sensor='IMX250',
                    input_shape=[3, 308, 257],
                    scene2mask=0.4,
                    mask2sensor=0.002,
                    target_dim=[164, 128],
                    do_optical=False,
                    requires_grad=True,
                    use_stn=False,
                    down='resize',
                    noise_type=None,
                    n_psf_mask=1),
                apply_affine=True,
                image_size=168),
            neck=dict(
                type='GlobalDepthWiseNeck',
                in_channels=384,
                out_channels=128,
                kernel_size=(11, 11)),
            head=dict(
                type='IdentityClsHead',
                loss=dict(type='ArcMargin', out_features=93955)),
            init_cfg=dict(
                type='Pretrained',
                checkpoint=
                '/root/caixin/RawSense/nolens_mmcls/logs/a_no_optical_face/full_with_base/epoch_50.pth'
            )),
        teacher_trainable=False,
        teacher_norm_eval=True,
        components=[
            dict(
                student_module='neck.fc',
                teacher_module='neck.fc',
                losses=[
                    dict(
                        type='DistanceWiseRKD',
                        name='distance_wise_loss',
                        loss_weight=100.0,
                        with_l2_norm=True),
                    dict(
                        type='AngleWiseRKD',
                        name='angle_wise_loss',
                        loss_weight=200.0,
                        with_l2_norm=True)
                ])
        ]))
work_dir = 'logs/distill/face/vit2optical'
auto_resume = False
gpu_ids = range(0, 8)
