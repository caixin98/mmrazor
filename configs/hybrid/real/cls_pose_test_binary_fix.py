
_base_ = [
    '../../_base_/datasets/face/celeb_propagate_test_bg_updater_rotate_scale_shift.py',
]


test_pipeline = [
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
            # dict(type='TorchAffineRTS', translate = (0.2,0.2),
                #  scale_factor=0.2, prob=1.0),
            dict(type="TorchAffineRTS",
                translate = (0.2,0.2),
                angle=(0, 0),
                return_translate=True,
                scale_factor=0.2,
                prob=0.0),
            # dict(type='AddBackground', img_dir='/mnt/workspace/RawSense/data/BG-20k/testval',size = (100, 100)),
            dict(type='ToTensor', keys=['fold', 'label']),
            dict(
                type='StackImagePair',
                keys=['img1_nopad', 'img1_flip_nopad', 'img2_nopad', 'img2_flip_nopad'],
                out_key='img'),
            dict(type='Collect', keys=['img', 'fold', 'label', 'affine_matrix'])
        ]

data = dict(
    test=dict(
    workers_per_gpu=4,
        type='LFW',
            load_pair = False,
            use_flip = False,
            img_prefix='/mnt/workspace/RawSense/data/lfw/lfw-112X96',
            pair_file='/mnt/workspace/RawSense/data/lfw/pairs.txt',
            pipeline=test_pipeline
   ),
    train_dataloader=dict(samples_per_gpu=160, persistent_workers=False),
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(workers_per_gpu=2,samples_per_gpu=64))



cls_checkpoint = "logs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary_fix/latest.pth"
pose_checkpoint = "logs/pose_with_cls_data/celeb_bg_cls_psf_fix/latest.pth"

cls_optical = dict(
    type='SoftPsfConv',
    feature_size=2.76e-05,
    sensor='IMX250',
    input_shape=[3, 308, 257],
    scene2mask=0.4,
    mask2sensor=0.002,
    target_dim=[164, 128],
    center_crop_size=[240, 200],
    requires_grad=True,
    use_stn=False,
    down="resize",
    noise_type="gaussian",
    do_affine = True,
    binary=True,
    do_optical=False,
    n_psf_mask=1)

cls_model = dict(
    type = 'mmcls.AffineFaceImageClassifier',
    backbone=dict(
        type='T2T_ViT_optical',
        optical=cls_optical,
        image_size=168),
    neck=dict(
        type='GlobalDepthWiseNeck',
        in_channels=384,
        out_channels=128,
        kernel_size=(11, 11)),
    head=dict(
        type='IdentityClsHead',
        loss=dict(type='ArcMargin', out_features=93955)))

pose_optical = dict(
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
    # load_weight_path="/root/caixin/RawSense/mmrazor/logs/distill/face/vit2optical_bg_updater_rotate/epoch_90.pth",
    binary=True,
    requires_grad_psf=False,
    do_optical=False,
    n_psf_mask=1)
pose_model = model = dict(
    type='TopDown',
   backbone=dict(
        type='T2T_ViT_optical',
        optical=pose_optical,
        image_size=168),
    # backbone=dict(
    #     type='T2T_ViT',
    #     img_size=64),
    neck=dict(type='GlobalAveragePooling'),
    keypoint_head=dict(
        type='DeepposeRegressionHead',
        in_channels=384,
        num_joints=1,
        loss_keypoint=dict(type='SmoothL1Loss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(flip_test = False))