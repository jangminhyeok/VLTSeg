import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
from mmengine.model import revert_sync_batchnorm
from mmengine import Config
from mmpretrain.models.utils import (RotaryEmbeddingFast, SwiGLUFFN, build_norm_layer, resize_pos_embed)
import xformers.ops as xops
import vltseg
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

# This sets the default registry to mmseg::model (instead of mmengine::model), which is where SegDataPreProcessor is found
# This is not necessary when using the test script, since default_runtime.py includes default_scope = 'mmseg'
# In this file, we are only building the 'model' part of the config, so that information is lost
register_all_modules(True)

def rsz(input, l):
    transform = torchvision.transforms.Resize((l, l))
    return transform(input)

config = "configs/demo_composition_config.py"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MODELS.build(Config.fromfile(config).model)
model.to(device)
model.init_weights()
if device == 'cpu':
    model = revert_sync_batchnorm(model)

torch.no_grad()

image = torch.load("images/CLIP_resized.pt", map_location=device)
#result = model.backbone.forward(image)[3]
#result /= result.norm(dim=-1, keepdim=True)
#other_result = torch.load("/work/kuehl/Projects/EVA/EVA-CLIP/rei/eva-clip_prehead_features_raw.pt", map_location=device)
#other_x = torch.load("/work/kuehl/Projects/EVA/EVA-CLIP/rei/other_x.pt", map_location=device)
self = model.backbone
x = image

B = x.shape[0]
x, patch_resolution = self.patch_embed(x)

if self.cls_token is not None:
    # stole cls_tokens impl from Phil Wang, thanks
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

x = x + resize_pos_embed(
    self.pos_embed,
    self.patch_resolution,
    patch_resolution,
    mode=self.interpolate_mode,
    num_extra_tokens=self.num_extra_tokens)
x = self.drop_after_pos(x)

x = self.pre_norm(x)

outs = []

#################################

self = self.layers[0]
old_x = x

x = self.norm1(x)                      # matches
#x = self.attn(x, patch_resolution)    # does not match, save and copy over
x = torch.load('/work/kuehl/Projects/EVA/EVA-CLIP/rei/initial_x.pt', map_location=device)
x = self.drop_path(x)
#x = old_x + x

