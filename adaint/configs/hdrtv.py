exp_name = 'ailut_hdrtv'

custom_imports=dict(
    imports=['adaint'],
    allow_failed_imports=False)

# model settings
model = dict(
    type='AiLUT',
    n_ranks=5,
    n_vertices=33,
    en_adaint=True,
    en_adaint_share=False,
    backbone='res18', # 'tpami'
    pretrained=True,
    n_colors=3,

    # Reconstruction loss
    recons_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),

    # Perceptual loss for better visual quality and smoother transitions
    #perceptual_loss=dict(
    #    type='PerceptualLoss',
    #    layer_weights={'4': 1.0, '9': 1.0, '18': 1.0},
    #    vgg_type='vgg19',
    #    use_input_norm=True,
    #    perceptual_weight=0.1,
    #    style_weight=0.0,
    #    norm_img=False,
    #    criterion='l1'
    #),

    # Gradient loss for smoother color transitions
    gradient_loss=dict(
        type='GradientLoss',
        loss_weight=0.1
    ),

    # Color gamut loss to expand HDR color range
    gamut_loss=dict(
        type='ColorGamutLoss',
        loss_weight=0.05,
        num_bins=64,
        color_space='rgb'
    ),

    # HDR tone mapping loss to preserve HDR characteristics
    hdr_tone_loss=dict(
        type='HDRToneLoss',
        loss_weight=0.2,
        epsilon=1e-6
    ),

    # Edge-aware weighting to reduce blocky artifacts
    edge_aware_weight=0.1,

    # Regularization factors
    sparse_factor=0.0001,
    smooth_factor=0.2,  # Enabled for LUT smoothness
    monotonicity_factor=10.0
)
# model training and testing settings
train_cfg = dict(n_fix_iters=3329*5)
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_dataset_type = 'HDRTV1K'
val_dataset_type = 'HDRTV1K'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        backend='pillow',
        channel_order='rgb'),
    dict(type='FlipChannels', keys=['lq']), # BGR->RGB
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        backend='cv2',
        flag='unchanged'),
    dict(type='RandomRatioCrop', keys=['lq', 'gt'], crop_ratio=(0.6, 1.0)),
    dict(type='Resize', keys=['lq', 'gt'], scale=(448, 448), backend='cv2', interpolation='bilinear'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='FlexibleRescaleToZeroOne', keys=['lq', 'gt'], precision=32),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        backend='pillow',
        channel_order='rgb'),
    dict(type='FlipChannels', keys=['lq']), # BGR->RGB
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        backend='cv2',
        flag='unchanged'),
    dict(type='FlexibleRescaleToZeroOne', keys=['lq', 'gt'], precision=32),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path'])
]

data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=16),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type=train_dataset_type,
        dir_lq='/home/sas/ke/dataset/train_sdr',
        dir_gt=f'/home/sas/ke/dataset/train_hdr',
        ann_file='/home/sas/ke/AdaInt/adaint/annfiles/HDRTV1K/train.txt',
        pipeline=train_pipeline,
        test_mode=False,
        filetmpl_lq='{}.png',
        filetmpl_gt='{}.png'),
    # val
    val=dict(
        type=val_dataset_type,
        dir_lq='/home/sas/ke/dataset/train_sdr',
        dir_gt=f'/home/sas/ke/dataset/train_hdr',
        ann_file='/home/sas/ke/AdaInt/adaint/annfiles/HDRTV1K/val.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}.png',
        filetmpl_gt='{}.png'),
    # test
    test=dict(
        type=val_dataset_type,
        dir_lq='/home/sas/ke/dataset/test_sdr',
        dir_gt=f'/home/sas/ke/dataset/test_hdr',
        ann_file='/home/sas/ke/AdaInt/adaint/annfiles/HDRTV1K/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}.png',
        filetmpl_gt='{}.png'),
)

# optimizer
optimizers = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=0,
    betas=(0.9, 0.999),
    eps=1e-8,
    paramwise_cfg=dict(custom_keys={'adaint': dict(lr_mult=0.1)}))
lr_config = None

# learning policy
total_iters = 3329*200

checkpoint_config = dict(interval=3329, save_optimizer=True, by_epoch=False)
evaluation = dict(
    interval=33290,
    save_image=True
)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
