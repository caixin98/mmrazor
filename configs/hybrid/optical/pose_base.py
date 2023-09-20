from mmcv import Config
_cls_base_ = Config.fromfile('configs/recog/full_gaussian.py')
_pose_base_ = Config.fromfile('configs/pose/retina_wflw5.py')

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(
        pose=dict(samples_per_gpu=_pose_base_.data.train_dataloader.samples_per_gpu),
    ),
    train=dict(
        pose=_pose_base_.data.train,
    ),
    val_dataloader=dict(samples_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=2),
    val=dict(
        type='HybridDataset',
        pose_dataset=_pose_base_.data.val,
        test_mode=True
    ),
    test=dict(
        type='HybridDataset',
        pose_dataset=_pose_base_.data.test,
        test_mode=True
    ),
)

model = dict(
    type='BaseHybrid',
    image_size=168,
    optical=_cls_base_.optical,
    posenet=_pose_base_.model,
)

# optimizer
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True


optimizer = dict(type='AdamW',lr=5e-4, weight_decay=0.05)
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

del Config, _cls_base_, _pose_base_