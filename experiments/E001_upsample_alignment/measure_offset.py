import torch, segmentation_models_pytorch as smp
from manyseg import create_model

def offsets(model, H=256, pts=None, reps=2):
    model.eval(); res=[]
    pts = pts or [(i, j) for i in range(96, 160, 7) for j in range(96, 160, 11)]
    for (i,j) in pts:
        g=0
        for r in range(reps):
            x=torch.randn(1,3,H,H,requires_grad=True)
            model(x)[0,:,i,j].sum().backward(); g=g+x.grad.abs().sum((0,1))
        ys=torch.arange(H,dtype=g.dtype)[:,None]; xs=torch.arange(H,dtype=g.dtype)[None]
        res.append(((g*ys).sum()/g.sum()-i, (g*xs).sum()/g.sum()-j))
    t=torch.tensor(res); return t.mean(0).tolist(), t.std(0).tolist()

torch.manual_seed(0)
for seed in range(2):
    torch.manual_seed(seed)
    print('manyseg', offsets(create_model(backbone='resnet18',pretrained=False,num_classes=3)))
    torch.manual_seed(seed)
    print('smp    ', offsets(smp.Unet('resnet18',encoder_weights=None,classes=3)))
