# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_eva_clip(ckpt):
    new_ckpt = OrderedDict()

    banned = {
        'mask_token',       # not found in file
        'lm_head.weight',   # not found in file
        'lm_head.bias',     # not found in file
        'visual.head'       # found, but not required for the featmaps that we need
    }

    for key, value in list(ckpt.items()):
        key_list = key.split('.')
        if any([ban in key for ban in banned]):
            continue

        if key_list[0] == 'visual':
            key_list[0] = 'backbone'

            # These keys are missing in my mmpretrain.eva02 checkpoint.
            # In fact, when loading the checkpoint, the model calls these parameters unexpected and fails to take them.
            # These keys represent a fully connected layer after the class token output, used for simple classification.
            # This is unnecessary when using a full, proper task head
            if key_list[1] == 'head':
                key_list[1] = 'head.fc'

            elif key_list[1] == 'patch_embed':
                if key_list[2] == 'proj':
                    key_list[2] = 'projection'

            # These keys are only needed for the flag final_norm
            elif key_list[1] == 'norm':
                key_list[1] = 'ln1'

            # These keys are missing in the eva-clip checkpoint, though their corresponding layer is implemented
            # They are only needed for the output type "avg_featmap"
            elif key_list[1] == 'fc_norm':
                key_list[1] = 'ln2'

            elif key_list[1] == 'blocks':
                key_list[1] = 'layers'
                # In this case, key_list[2] is the layer number from 0 to 23.
                # Therefore, we continue with...

                if key_list[3] == 'mlp':
                    if key_list[4] == 'w1':
                        # For base and large version, mlp is implemented with
                        # 2 linears, where w1 and w2 need to be concatenated
                        # into a new key w12.(weight/bias)
                        key_list[4] = 'w12'

                        key_w2 = key.replace('w1', 'w2')
                        value_w2 = ckpt[key_w2]
                        value = torch.cat((value, value_w2))

                    elif key_list[4] == 'w2':
                        # We handle w2 above, by concatenating with w1
                        continue

                    elif key_list[4] == 'ffn_ln':
                        key_list[4] = 'norm'

                elif key_list[3] == 'attn':
                    # If the next key is "rope" or "norm", we let it pass

                    if key_list[4] == 'q_proj':
                        # For base and large version, qkv projection is
                        # implemented with three linear layers, which we concat
                        key_list[4] = 'qkv'

                        key_k_proj = key.replace('q_proj', 'k_proj')
                        key_v_proj = key.replace('q_proj', 'v_proj')
                        value_k_proj = ckpt[key_k_proj]
                        value_v_proj = ckpt[key_v_proj]
                        value = torch.cat((value, value_k_proj, value_v_proj))

                    elif key_list[4] in ['k_proj', 'v_proj']:
                        # We handle k_proj and v_proj above
                        continue

                    elif key_list[4] == 'q_bias':
                        # Same as q/k/v_proj, but k_bias is always 0
                        key_list[4] = 'qkv.bias'

                        key_v_bias = key.replace('q_bias', 'v_bias')
                        value_k_bias = torch.zeros_like(value, requires_grad=False)
                        value_v_bias = ckpt[key_v_bias]
                        value = torch.cat((value, value_k_bias, value_v_bias))

                    elif key_list[4] in ['k_bias', 'v_bias']:
                        # We handle k_bias and v_bias above
                        # Though k_bias really shouldn't exist
                        continue

                    elif key_list[4] == 'inner_attn_ln':
                        key_list[4] = 'inner_attn_norm'
        else:
            # This catches both the entirety of the text transformer, which
            # starts with 'text.', and global parameters like logit_scale.
            # We do not want to copy them over to the backbone state_dict.
            continue
        
        new_key = '.'.join(key_list)
        new_ckpt[new_key] = value

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained eva02 '
        'models to mmpretrain style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'module' in checkpoint:
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint

    weight = convert_eva_clip(state_dict)
    new_checkpoint = {'state_dict': weight }  # 'meta': {'mmpretrain_version': '1.0.0rc7', 'dataset_meta': {}}
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(new_checkpoint, args.dst)


if __name__ == '__main__':
    main()