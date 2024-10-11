_base_ = [
    '_base_/models/mask2former.py',
    '_base_/datasets/train/gta_512x512.py',
    '_base_/datasets/test/synth2cityscapes.py',
    '_base_/schedules/schedule_1k_frozen.py',
    '_base_/default_runtime.py'
]

crop_size = (512, 512)
stride_size = (426,426)
pretrained = '/work/kuehl/checkpoints/converted_EVA02_CLIP_L_psz14_s4B.pth'
num_gpus = 2
num_samples_per_gpu_train = 8
num_workers_per_gpu_train = 1
num_samples_per_gpu_test = 2
num_workers_per_gpu_test = 1

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(size=crop_size),
    backbone=dict(
        type='ViTEVA02WithXAttn',
        arch='l',
        img_size=512,
        patch_size=14,
        drop_path_rate=0.15, # Does this have a positive impact?
        xattn=True,
        sub_ln=True,
        final_norm=False,
        out_indices=[9,14,19,23],
        norm_cfg=dict(
            type='LN',
            eps=1e-06,
        ),
        out_type='featmap',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone.',
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