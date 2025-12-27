exp_name = 'ailut_hdrtv_spatial_offset'

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
    smooth_factor=0.005,               # 轻微平滑正则，防止 LUT 过于精细
    monotonicity_factor=10.0,
    # === 空间自适应偏移配置 ===
    # offset 可以实现两种效果：
    # 1. 发散：相同颜色 → 不同输出（解决边缘发灰）
    # 2. 收敛：不同颜色（噪声）→ 相同输出（解决块状噪声）
    en_spatial_offset=True,            # 启用空间偏移
    offset_hidden=16,                  # 偏移生成器隐藏层维度
    offset_scale_factor=1,             # 全分辨率处理，保留块边界细节
    offset_scale=0.1,                  # 偏移范围 ±0.1
    offset_mode='input',               # 作用于输入，更稳定
    # === 关闭 pre/post refine ===
    en_pre_refine=False,
    en_post_refine=False,
    en_input_smooth=False,             # 不用显式平滑，让 offset 自己学
    refine_hidden=16,
    refine_scale_factor=4,
    pre_refine_residual_scale=0.1,
    post_refine_residual_scale=0.1,
    residual_reg_factor=0.0,
    recons_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'))

# model training and testing settings
train_cfg = dict(
    n_fix_iters=77 * 5,              # AdaInt 冻结 5 个 epoch
    n_fix_refine_iters=0             # 不冻结（spatial_offset 从头训练）
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
        'backbone': dict(lr_mult=0.5),       # backbone 慢一点
        'lut_generator': dict(lr_mult=0.5),  # LUT 慢一点
        'adaint': dict(lr_mult=0.1),
        'spatial_adaptive': dict(lr_mult=3.0),  # offset 快 3 倍
    }))
lr_config = None

# learning policy
# 总共 500 epoch: 前 50 epoch 只训练 LUT，后 450 epoch 联合训练
total_iters = 77 * 800

checkpoint_config = dict(interval=7700, save_optimizer=True, by_epoch=False)  # 每 100 epoch 保存一次
evaluation = dict(interval=770, save_image=True, save_best=True, key_indicator='PSNR')  # 每 10 个 epoch 评估，保存最佳 PSNR
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
