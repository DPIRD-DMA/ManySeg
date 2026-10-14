import timm, torch, sys, time, traceback
from torch.export import Dim
torch.set_grad_enabled(False)
which = sys.argv[1]; mode = sys.argv[2]
def build():
    if which == 'resnet34': return timm.create_model('resnet34', pretrained=False, features_only=True), (256,320), (257,301)
    if which == 'convnext': return timm.create_model('convnext_tiny', pretrained=False, features_only=True), (256,320), (257,301)
    if which == 'effb0': return timm.create_model('efficientnet_b0', pretrained=False, features_only=True), (256,320), (257,301)
    if which == 'vit':
        m = timm.create_model('vit_small_patch16_224', pretrained=False, dynamic_img_size=True, features_only=True, out_indices=(2,5,8,11)); return m, (256,320), (224,288)
    if which == 'vitpad':
        m = timm.create_model('vit_small_patch16_224', pretrained=False, dynamic_img_size=True, dynamic_img_pad=True, features_only=True, out_indices=(2,5,8,11)); return m, (256,320), (257,301)
    if which == 'swin':
        m = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, strict_img_size=False, features_only=True); return m, (256,320), (224,288)
m, s1, s2 = build(); m.eval()
def shp(o): return [tuple(t.shape[2:]) if t.shape[1]>t.shape[-1]*0 else 0 for t in o]
t0=time.time()
try:
    if mode.startswith('compile'):
        dev = 'cuda'; m = m.to(dev)
        cm = torch.compile(m, fullgraph=True, dynamic=True)
        a = cm(torch.randn(1,3,*s1,device=dev)); b = cm(torch.randn(1,3,*s2,device=dev))
        ref = m(torch.randn(1,3,*s2,device=dev))
        print(which, mode, 'OK', [tuple(t.shape) for t in b][-1], 'match', all(x.shape==y.shape for x,y in zip(b,ref)), f'{time.time()-t0:.0f}s')
    else:
        if mode == 'export_free':
            H = Dim('H', min=64, max=2048); W = Dim('W', min=64, max=2048)
        else:  # multiple of 32
            H = 32*Dim('h', min=2, max=64); W = 32*Dim('w', min=2, max=64)
        ep = torch.export.export(m, (torch.randn(1,3,*s1),), dynamic_shapes=({2:H, 3:W},))
        out = ep.module()(torch.randn(1,3,*s2))
        print(which, mode, 'OK', [tuple(t.shape) for t in out], f'{time.time()-t0:.0f}s')
except Exception as e:
    msg = str(e).replace('\n',' | ')
    print(which, mode, 'FAIL', type(e).__name__, msg[:700])
