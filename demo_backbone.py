import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
from mmengine.model import revert_sync_batchnorm
from mmengine import Config
import vltseg
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules

register_all_modules(True) # This sets the default registry to mmseg::model (instead of mmengine::model), which is where SegDataPreProcessor is found

def extract(img_path, load=True):
    if load:
        image = read_image(img_path, mode=ImageReadMode.RGB).to(torch.float32).to(device).unsqueeze(0)
        transform = torchvision.transforms.Resize((224, 224))
        image = transform(image)
    else:
        image = torch.load(img_path, map_location=device)

    with torch.no_grad():
        image_features = model.extract_feat(image)[0]
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features


config = "configs/demo_backbone_config.py"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MODELS.build(Config.fromfile(config).model)
model.to(device)
model.init_weights()
if device == 'cpu':
    model = revert_sync_batchnorm(model)

# The extract method: Loads the image, applies the preprocessor and then applies the forward step in test mode
result = extract("images/CLIP_resized.pt", load=False)
other_result = torch.load('images/eva-clip_prehead_features.pt', map_location=device)