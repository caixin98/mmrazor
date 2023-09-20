dataset_info = dict(
    dataset_name='retinaface',
    paper_info=dict(
        author='Deng, Jiankang and Guo, Jia and Ververas, Evangelos and '
        'Kotsia, Irene and Zafeiriou, Stefanos',
        title='RetinaFace: Single-stage Dense Face Localisation in the Wild',
        container='arXiv:1905.00641',
        year='2019',
    ),
    keypoint_info={
        0: dict(name='left_eye', id=0, color=[255, 255, 255], type='', swap='right_eye'),
        1: dict(name='right_eye', id=1, color=[255, 0, 255], type='', swap='left_eye'),
        2: dict(name='nose', id=2, color=[0, 255, 255], type='', swap=''),
        3: dict(name='mouth_left', id=3, color=[0, 0, 255], type='', swap='mouth_right'),
        4: dict(name='mouth_right', id=4, color=[255, 0, 0], type='', swap='mouth_left'),
    },
    skeleton_info={},
    joint_weights=[1.] * 5,
    sigmas=[])  

evaluation = dict(metric=['NME'], save_best='NME')

channel_cfg = dict(
    num_output_channels=5,
    dataset_joints=5,
    dataset_channel=[
        list(range(5)),
    ],
    inference_channel=list(range(5)))

# model settings
model = dict(
    type='TopDown',
   backbone=dict(
        type='T2T_ViT'),
    neck=dict(type='GlobalAveragePooling'),
    keypoint_head=dict(
        type='DeepposeRegressionHead',
        in_channels=384,
        num_joints=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='SmoothL1Loss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(flip_test=True))

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
        type='TopDownGetRandomScaleRotation', rot_factor=30,
        scale_factor=0.4),
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
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
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
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs']),
]
test_pipeline = val_pipeline

data_root = '/mnt/workspace/RawSense/data/retina_single_face_coco'
image_root = '/mnt/workspace/RawSense/data/widerface/WIDER_train'
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=64),
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='FaceRetinaDataset',
        ann_file=f'{data_root}/face_landmarks_retina_train.json',
        img_prefix=f'{image_root}/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info=dataset_info),
    val=dict(
        type='FaceRetinaDataset',
        ann_file=f'/mnt/workspace/RawSense/data/wflw/annotations_five_point/face_landmarks_wflw_test.json',
        img_prefix=f'/mnt/workspace/RawSense/data/wflw/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info=dataset_info),
    test=dict(
        type='FaceRetinaDataset',
        ann_file=f'/mnt/workspace/RawSense/data/wflw/annotations_five_point/face_landmarks_wflw_test.json',
        img_prefix=f'/mnt/workspace/RawSense/data/wflw/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info=dataset_info),
)

