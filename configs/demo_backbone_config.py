_base_ = [
    './_base_/models/segformer_mit-b0.py', './_base_/datasets/cityscapes.py',
    './_base_/default_runtime.py', './_base_/schedules/schedule_160k.py'
]

data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    # convert image from BGR to RGB
    bgr_to_rgb=True,
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='ViTEVA02WithXAttn',
        arch='l',
        img_size=224,
        patch_size=14,
        xattn=True,
        sub_ln=True,
        final_norm=True,
        norm_cfg=dict(
            type='LN',
            eps=1e-06,
        ),
        out_type='cls_token',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/work/kuehl/checkpoints/converted_EVA02_CLIP_L_psz14_s4B.pth',
            prefix='backbone.',
        ),
    )
)

