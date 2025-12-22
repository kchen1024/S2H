exp_name = 'ailut_hdrtv_refine'

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
    backbone='res18',
    pretrained=True,
    n_colors=3,
    sparse_factor=0.0001,
    smooth_factor=0,
    monotonicity_factor=10.0,
    # === Refinement 配置 ===
    en_refine=True,                    # 启用 refinement
    refine_hidden=24,                  # 隐藏层维度
    refine_use_backbone_feat=False,    # 是否复用 backbone 特征
    mask_sparse_factor=0.05,           # mask 稀疏正则
    residual_reg_factor=0.005,         # 残差幅度约束
    refine_smooth_factor=0.0,          # 梯度平滑 (可选)
    recons_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'))

# model training and testing settings
train_cfg = dict(
    n_fix_iters=3329*5,           # AdaInt 冻结 5 个 epoch
    n_fix_refine_iters=3329*10    # Refinement 冻结 10 个 epoch (让 LUT 先学)
)
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
    paramwise_cfg=dict(custom_keys={
        'adaint': dict(lr_mult=0.1),
        'refine_head': dict(lr_mult=0.1)  # refinement 用小学习率
    }))
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
resume_from = './work_dirs/ailut_hdrtv/iter_359532.pth'
workflow = [('train', 1)]
find_unused_parameters = True
