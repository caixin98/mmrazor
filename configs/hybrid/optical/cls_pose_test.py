
_base_ = [
    '../../_base_/datasets/face/celeb_propagate_rotate_shift.py',
]
cls_checkpoint = "/root/caixin/RawSense/mmrazor/logs/distill/face/vit2optical_bg_af_updater_rotate_shift_crop/latest.pth"
pose_checkpoint = "/root/caixin/RawSense/nolens_face_align/logs/a_optical_face/celeb_bg_cls_psf/best_NME_epoch_114.pth"

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
    # binary=True,
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
    load_weight_path="/root/caixin/RawSense/mmrazor/logs/distill/face/vit2optical_bg_updater_rotate/epoch_90.pth",
    requires_grad_psf=False,
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