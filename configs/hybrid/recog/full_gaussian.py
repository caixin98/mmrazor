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
    # do_affine=True,
    n_psf_mask=1)

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
                output_dim=[308, 257, 3],
               ),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
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
                output_dim=[308, 257, 3],
                ),
            dict(type='ToTensor', keys=['fold', 'label']),
            dict(
                type='StackImagePair',
                keys=['img1', 'img1_flip', 'img2', 'img2_flip'],
                out_key='img'),
            dict(type='Collect', keys=['img', 'fold', 'label'])
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
                output_dim=[308, 257, 3],
                ),
            # dict(type='TorchAffineRTS', translate = (0.2,0.2),
                #  scale_factor=0.2, prob=1.0),
            dict(type="TorchAffineRTS",angle=(0,0),
                translate = (0.2,0.2),
                # scale_factor=0.2,
                prob=0.0),
            dict(type='ToTensor', keys=['fold', 'label']),
            dict(
                type='StackImagePair',
                keys=['img1', 'img1_flip', 'img2', 'img2_flip'],
                out_key='img'),
            dict(type='Collect', keys=['img', 'fold', 'label'])
        ]),
    train_dataloader=dict(samples_per_gpu=100),
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64))

model = dict(
    type='FaceImageClassifier',
    backbone=dict(
        type='T2T_ViT'),
    neck=dict(
        type='GlobalDepthWiseNeck',
        in_channels=384,
        out_channels=128,
        kernel_size=(11, 11)),
    head=dict(
        type='IdentityClsHead',
        loss=dict(type='ArcMargin', out_features=93955)))

evaluation = dict(metric='accuracy')