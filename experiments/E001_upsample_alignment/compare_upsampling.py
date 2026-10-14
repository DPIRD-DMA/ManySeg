import torch, torch.nn.functional as F, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from measure_offset import offsets
import manyseg.decoders.unet as U, manyseg.heads.heads as H
from manyseg import create_model

def aligned_up(x, size):
    # coarse sample j sits at fine coordinate 2j (conv 'same' padding convention)
    h, w = x.shape[-2:]
    xp = F.pad(x, (0, 1, 0, 1), mode="replicate")
    y = F.interpolate(xp, size=(2*h+1, 2*w+1), mode="bilinear", align_corners=True)
    return y[..., :size[0], :size[1]]

orig = U.UNetBlock.forward
def fwd(self, x, skip):
    if skip is not None:
        x = torch.cat([aligned_up(x, skip.shape[-2:]), skip], 1)
    else:
        x = aligned_up(x, (2*x.shape[-2], 2*x.shape[-1]))
    return self.conv2(self.conv1(x))
U.UNetBlock.forward = fwd
for seed in range(2):
    torch.manual_seed(seed); print('aligned-bilinear', offsets(create_model(backbone='resnet18',pretrained=False,num_classes=3)))
U.UNetBlock.forward = orig
# bilinear align_corners=False (common alternative)
def fwd2(self, x, skip):
    size = skip.shape[-2:] if skip is not None else (2*x.shape[-2], 2*x.shape[-1])
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    if skip is not None: x = torch.cat([x, skip], 1)
    return self.conv2(self.conv1(x))
U.UNetBlock.forward = fwd2
torch.manual_seed(0); print('bilinear-acF', offsets(create_model(backbone='resnet18',pretrained=False,num_classes=3)))
