exp_name = 'ailut_hdrtv_stage2'

custom_imports = dict(
    imports=['adaint'],
    allow_failed_imports=False)

# model settings - 冻结 LUT，只训练 deblocking
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
    smooth_factor=0.0,
    monotonicity_factor=10.0,
    # === 关闭其他模块 ===
    en_spatial_offset=False,
    en_pre_refine=False,
    en_post_refine=False,
    # === 启用 Deblocking ===
    en_deblocking=True,
    deblocking_hidden=24,
    deblocking_scale_factor=1,         # 全分辨率
    deblocking_residual_scale=0.15,
    en_input_smooth=False,
    residual_reg_factor=0.0,
    recons_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'))

# model training and testing settings
train_cfg = dict(
    n_fix_iters=0,
    n_fix_refine_iters=0,
    freeze_lut=True,                   # 冻结 LUT 相关模块
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
    train=dict(
        type=train_dataset_type,
        dir_lq='/home/sas/ke/dataset/train_sdr',
        dir_gt='/home/sas/ke/dataset/train_hdr',
        ann_file='/home/sas/ke/S2H-add_refinemodule/adaint/annfiles/HDRTV1K/train.txt',
        pipeline=train_pipeline,
        test_mode=False,
        filetmpl_lq='{}.png',
        filetmpl_gt='{}.png'),
    val=dict(
        type=val_dataset_type,
        dir_lq='/home/sas/ke/dataset/test_sdr',
        dir_gt='/home/sas/ke/dataset/test_hdr',
        ann_file='/home/sas/ke/S2H-add_refinemodule/adaint/annfiles/HDRTV1K/val.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}.png',
        filetmpl_gt='{}.png'),
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

# optimizer - 只优化 deblocking
optimizers = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=0,
    betas=(0.9, 0.999),
    eps=1e-8,
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.0),       # 冻结
        'lut_generator': dict(lr_mult=0.0),  # 冻结
        'adaint': dict(lr_mult=0.0),         # 冻结
        'deblocking': dict(lr_mult=1.0),
    }))
lr_config = None

total_iters = 77 * 200                 # 200 epoch

checkpoint_config = dict(interval=7700, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=770, save_image=True, save_best=True, key_indicator='PSNR')
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
visual_config = None

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
# 加载阶段1训练好的模型
load_from = './work_dirs/ailut_hdrtv_stage1/best_PSNR.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
