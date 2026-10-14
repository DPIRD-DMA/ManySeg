import torch
from torch.utils.flop_counter import FlopCounterMode
from fastai.vision.models.unet import DynamicUnet
from fastai.vision.learner import create_body
from torchvision.models import resnet34
import segmentation_models_pytorch as smp
fa = DynamicUnet(create_body(resnet34(weights=None),3,False,-2), 2, (256,256)).train()
sm = smp.Unet("resnet34", encoder_weights=None, classes=2).train()
x=torch.randn(2,3,256,256)
def per(m, parts):
    out={}
    for n,mod in parts:
        mod.register_forward_hook(lambda *a: None)
    fc=FlopCounterMode(display=False, depth=4)
    with fc: m(x).sum().backward()
    d=fc.get_flop_counts()
    for k,v in d.items():
        if k.count('.')<=2: print(k, round(sum(v.values())/2/1e9,2))  # per-sample, fwd+bwd
per(fa,[]); per(sm,[])
import time
for name,m in (("fa",fa),("sm",sm)):
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        mm=m.cuda(); xx=torch.randn(8,3,256,256).cuda()
        mm(xx).sum().backward(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t=time.time()
        for _ in range(10): mm(xx).sum().backward()
        torch.cuda.synchronize(); print(name,"bs8 fwd+bwd s/it",(time.time()-t)/10,"peakMB",torch.cuda.max_memory_allocated()/2**20)
