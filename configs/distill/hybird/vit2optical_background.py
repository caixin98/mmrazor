from mmcv import Config
_cls_base_ = Config.fromfile('configs/hybrid/recog/full_gaussian.py')
teacher_ckpt = "/root/caixin/RawSense/nolens_mmcls/logs/a_no_optical_face/full_with_base/epoch_50.pth"
optical = dict(
    type='SoftPsfConv',
    feature_size=2.76e-05,
    sensor='IMX250',
    input_shape=[3, 308, 257],
    scene2mask=0.4,
    mask2sensor=0.002,
    target_dim=[164, 128],
    requires_grad=True,
    use_stn=False,
    down="resize",
    noise_type="gaussian",
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
    down="resize",
    noise_type=None,
    n_psf_mask=1)

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(
        cls=dict(samples_per_gpu=_cls_base_.data.train_dataloader.samples_per_gpu),
    ),
    train=dict(
        cls=_cls_base_.data.train,
    ),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    val=dict(
        type='HybridDataset',
        cls_dataset=_cls_base_.data.val,
        test_mode=True
    ),
    test=dict(
        type='HybridDataset',
        cls_dataset=_cls_base_.data.test,
        test_mode=True
    ),
)

cls_student = dict(
    type = 'mmcls.AffineFaceImageClassifier',
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

cls_teacher = dict(
    type = 'mmcls.AffineFaceImageClassifier',
    backbone=dict(
        type='T2T_ViT'),
    neck=dict(
        type='GlobalDepthWiseNeck',
        in_channels=384,
        out_channels=128,
        kernel_size=(11, 11)),
    head=dict(
        type='IdentityClsHead',
        loss=dict(type='ArcMargin', out_features=93955)),
    init_cfg=dict(type='Pretrained', checkpoint=teacher_ckpt),
)

student = dict(
    type='BaseHybrid',
    img_size=168,
    optical=optical,
    classifier=cls_student,
)

teacher = dict(
    type='BaseHybrid',
    img_size=168,
    optical=no_optical,
    classifier=cls_teacher,
    remove_bg=True,
)

algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='HybridArchitecture',
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
                student_module='classifier.neck.fc',
                teacher_module='classifier.neck.fc',
                  losses=[
                    dict(
                        type='DistanceWiseRKD',
                        name='distance_wise_loss',
                        loss_weight=00.0,
                        with_l2_norm=True),
                    dict(
                        type='AngleWiseRKD',
                        name='angle_wise_loss',
                        loss_weight=00.0,
                        with_l2_norm=True),
                ])
        ]),
)
# optimizer
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True


optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=20000,
    warmup_ratio=0.25)
checkpoint_config = dict(by_epoch=False, interval=20000)
runner = dict(type='HybridIterBasedRunner', max_iters=200000)
evaluation = dict(interval=2000, cls_args=_cls_base_.evaluation)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
custom_hooks = [
    dict(type='VisualConvHook'),
    dict(type='VisualAfterOpticalHook'),
    # dict(type='BGUpdaterHook', max_progress = 0.2)
]
# custom_hooks = dict(_delete_=True)
del Config, _cls_base_