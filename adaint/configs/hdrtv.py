exp_name = 'ailut_hdrtv_refine'

custom_imports = dict(
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
    refine_hidden=16,                  # 隐藏层维度 (轻量)
    refine_scale_factor=4,             # 下采样倍数 (4K -> 960x540)
    residual_reg_factor=0.0,           # 不惩罚 residual
    recons_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'))

# model training and testing settings
train_cfg = dict(
    n_fix_iters=77 * 5,              # AdaInt 冻结 5 个 epoch (让 backbone 先稳定)
    n_fix_refine_iters=77 * 50       # RefineHead 冻结 50 个 epoch，让 LUT 先学好色彩映射
)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

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
    dict(type='FlipChannels', keys=['lq']),
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
    dict(type='FlipChannels', keys=['lq']),
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
        dir_gt='/home/sas/ke/dataset/train_hdr',
        ann_file='/home/sas/ke/S2H-add_refinemodule/adaint/annfiles/HDRTV1K/train.txt',
        pipeline=train_pipeline,
        test_mode=False,
        filetmpl_lq='{}.png',
        filetmpl_gt='{}.png'),
    # val
    val=dict(
        type=val_dataset_type,
        dir_lq='/home/sas/ke/dataset/test_sdr',
        dir_gt='/home/sas/ke/dataset/test_hdr',
        ann_file='/home/sas/ke/S2H-add_refinemodule/adaint/annfiles/HDRTV1K/val.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}.png',
        filetmpl_gt='{}.png'),
    # test
    test=dict(
        type=val_dataset_type,
        dir_lq='/home/sas/ke/dataset/test_sdr',
        dir_gt='/home/sas/ke/dataset/test_hdr',
        ann_file='/home/sas/ke/S2H-add_refinemodule/adaint/annfiles/HDRTV1K/test.txt',
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
        'refine_head': dict(lr_mult=1.0)  # 用正常学习率
    }))
lr_config = None

# learning policy
# 总共 300 epoch: 前 50 epoch 只训练 LUT，后 250 epoch 联合训练
total_iters = 77 * 300

checkpoint_config = dict(interval=7700, save_optimizer=True, by_epoch=False)  # 每 100 epoch 保存一次
evaluation = dict(interval=770, save_image=True)  # 每 10 个 epoch 评估一次
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
load_from = None      # 不加载预训练
resume_from = None    # 不恢复
workflow = [('train', 1)]
find_unused_parameters = True
