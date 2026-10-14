import torch, torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from fastai.vision.models.unet import DynamicUnet, UnetBlock
from fastai.vision.learner import create_body, _resnet_split
from torchvision.models import resnet34
import segmentation_models_pytorch as smp
torch.manual_seed(0)
np_ = lambda m: sum(p.numel() for p in m.parameters())
body = create_body(resnet34(weights=None), 3, False, -2)
print("body children:", [type(c).__name__ for c in body.children()])
fa = DynamicUnet(body, 2, (256,256)).eval()
print("fastai layers:")
x = torch.randn(1,3,256,256)
h=[]
for i,l in enumerate(fa.layers):
    l.register_forward_hook(lambda m,i_,o,i=i: h.append((i,type(m).__name__,tuple(o.shape), np_(m))))
with torch.no_grad(): out=fa(x)
for r in h: print(r)
enc_p = np_(fa.layers[0]); print("fastai enc params",enc_p,"decoder+head params",np_(fa)-enc_p, "total", np_(fa))
for i,l in enumerate(fa.layers):
    if isinstance(l,UnetBlock): print("UnetBlock",i,"shuf",l.shuf[0][0].weight.shape,"conv1",l.conv1[0].weight.shape,"conv2",l.conv2[0].weight.shape, "conv1 modules",[type(m).__name__ for m in l.conv1])
print("middle", fa.layers[3])
print("tail", fa.layers[-5:])
sm = smp.Unet("resnet34", encoder_weights=None, classes=2).eval()
h2=[]
for i,b in enumerate(sm.decoder.blocks):
    b.register_forward_hook(lambda m,i_,o,i=i: h2.append((i,tuple(o.shape),np_(m))))
with torch.no_grad(): out2=sm(x)
print("smp enc out_channels", sm.encoder.out_channels)
for r in h2: print("smp block",r)
print("smp enc",np_(sm.encoder),"dec",np_(sm.decoder),"head",np_(sm.segmentation_head),"total",np_(sm))
for name,m in (("fastai",fa),("smp",sm)):
    for sz in (256,):
        fc=FlopCounterMode(display=False)
        with fc, torch.no_grad(): m(torch.randn(1,3,sz,sz))
        tot=fc.get_total_flops()
        print(name,"GFLOPs(2*MAC)",tot/1e9)
# encoder-only flops
fc=FlopCounterMode(display=False)
with fc, torch.no_grad(): sm.encoder(x)
print("smp encoder GFLOPs", fc.get_total_flops()/1e9)
fc=FlopCounterMode(display=False)
with fc, torch.no_grad(): fa.layers[0](x)
print("fastai encoder GFLOPs", fc.get_total_flops()/1e9)
# non divisible sizes
for sz in (250, 257, 300):
    for name,m in (("fastai",fa),("smp",sm)):
        try:
            with torch.no_grad(): o=m(torch.randn(1,3,sz,sz))
            print(name,sz,"->",tuple(o.shape))
        except Exception as e: print(name,sz,"ERR",str(e)[:150])
# BN counts / bias in decoder
def cnt(m,t): return sum(isinstance(q,t) for q in m.modules())
print("fastai decoder BN count", sum(cnt(l,nn.BatchNorm2d) for l in fa.layers[1:]), "smp decoder BN", cnt(sm.decoder,nn.BatchNorm2d))
print("fastai skip bn init", fa.layers[4].bn.weight[:3], fa.layers[4].bn.bias[:3])
print("fastai splits sizes", [sum(p.numel() for p in g) for g in _resnet_split(fa)])
print("output std fastai", out.float().std().item(), "smp", out2.std().item())
