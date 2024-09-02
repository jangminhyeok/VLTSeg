import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
from mmengine.model import revert_sync_batchnorm
from mmengine import Config
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


with torch.no_grad():
    image = read_image('images/gray-dog-closeup.jpg', mode=ImageReadMode.RGB).to(torch.float32).to(device).unsqueeze(0)
    result = model.predict(image)
    predictions = result[0].seg_logits.data

print(str(result))