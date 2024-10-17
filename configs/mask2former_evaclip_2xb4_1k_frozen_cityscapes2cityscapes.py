_base_ = [
    '_base_/models/mask2former.py',
    '_base_/datasets/train/cityscapes_1024x1024.py',
    '_base_/datasets/test/real2cityscapes.py',
    '_base_/schedules/schedule_1k_frozen.py',
    '_base_/default_runtime.py'
]

crop_size = (1024, 1024)
stride_size = (768,768)
pretrained = '/work/kuehl/checkpoints/EVA02_CLIP_L_psz14_s4B.pt'
num_gpus = 2
num_samples_per_gpu_train = 4
num_workers_per_gpu_train = 1
num_samples_per_gpu_test = 2
num_workers_per_gpu_test = 1

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(size=crop_size),
    backbone=dict(
        type='EVA02',
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4*2/3,

        img_size=crop_size[0],
        in_chans=3,
        patch_size=14,
        out_indices=[9,14,19,23],

        qkv_bias=True,
        drop_path_rate=0.2,
        use_abs_pos_emb=True, 
        use_rel_pos_bias=False, 
        use_shared_rel_pos_bias=False,
        
        subln=True,
        xattn=True,
        naiveswiglu=True,
        rope=True,
        pt_hw_seq_len=16,
        intp_freq=True,

        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='visual.',
        ),
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=stride_size)
)

train_dataloader = dict(
    batch_size=num_samples_per_gpu_train,
    num_workers=num_workers_per_gpu_train,
)

val_dataloader = dict(
    batch_size=num_samples_per_gpu_test,
    num_workers=num_workers_per_gpu_test,
)

test_dataloader = dict(
    batch_size=num_samples_per_gpu_test,
    num_workers=num_workers_per_gpu_test,
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (# GPUs) x (# samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=num_gpus*num_samples_per_gpu_train)