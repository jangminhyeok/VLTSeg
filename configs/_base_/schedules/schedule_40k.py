iterations = 40000          # number of batches used in training
val_interval = 2000         # interval (number of iterations) for evaluation and checkpointing

# optimizer
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    # OpenMMLab applies the most specific (=longest) custom_keys first. If we simply used '.norm.' etc., then 'backbone' would be more specific.
    # Thus, we need to be roughly this specific to hit all norm layers in the backbone.
    # The alternative is to abbreviate 'backbone' by 'backb'.
    '.norm.weight': backbone_norm_multi,
    '.norm.bias': backbone_norm_multi,
    '.norm1.weight': backbone_norm_multi,
    '.norm1.bias': backbone_norm_multi,
    '.norm2.weight': backbone_norm_multi,
    '.norm2.bias': backbone_norm_multi,
    '.inner_attn_norm.weight': backbone_norm_multi,
    '.inner_attn_norm.bias': backbone_norm_multi,
    'backbone.pos_embed': backbone_embed_multi,
    'decode_head.query_embed': embed_multi,
    'decode_head.query_feat': embed_multi,
    'decode_head.level_embed': embed_multi
}

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0)
)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=500,
        end=iterations,
        by_epoch=False)
]

# training schedule for 5k iterations
train_cfg = dict(type='IterBasedTrainLoop', max_iters=iterations, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)