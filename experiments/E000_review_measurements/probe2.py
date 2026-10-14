import timm, torch
torch.set_grad_enabled(False)
SIZES = [(224,224),(256,320),(257,301),(97,131)]
def run(tag, m, sizes=SIZES):
    m.eval()
    for s in sizes:
        try:
            out = m(torch.randn(1,3,*s))
            print(tag, s, [tuple(o.shape[1:]) for o in out])
        except Exception as e:
            print(tag, s, 'FAIL', str(e).split('\n')[0][:150])
C = dict(pretrained=False, features_only=True)
run('vit dyn+pad', timm.create_model('vit_small_patch16_224', dynamic_img_size=True, dynamic_img_pad=True, **C))
run('eva dyn+pad', timm.create_model('eva02_tiny_patch14_224', dynamic_img_size=True, dynamic_img_pad=True, **C))
try: run('swin strict=False dyn_pad', timm.create_model('swin_tiny_patch4_window7_224', strict_img_size=False, dynamic_img_pad=True, **C))
except Exception as e: print('swin dyn_pad create FAIL', str(e)[:150])
run('swin img_size=(256,320)', timm.create_model('swin_tiny_patch4_window7_224', img_size=(256,320), **C), [(256,320)])
run('swin strict=False always_partition', timm.create_model('swin_tiny_patch4_window7_224', strict_img_size=False, always_partition=True, **C), [(256,320),(96,128)])
b = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False); b.set_input_size(img_size=(256,320), window_size=8)
print('swin set_input_size', [tuple(t.shape) for t in b.forward_intermediates(torch.randn(1,3,256,320), intermediates_only=True)])
b = timm.create_model('vit_small_patch16_224', pretrained=False); b.set_input_size(img_size=(256,320))
print('vit set_input_size', b.pos_embed.shape, [tuple(t.shape) for t in b.forward_intermediates(torch.randn(1,3,256,320), indices=[3,7,11], intermediates_only=True)])
print('vit fi output_fmt NLC', [tuple(t.shape) for t in b.forward_intermediates(torch.randn(1,3,256,320), indices=2, output_fmt='NLC', intermediates_only=True)])
run('maxvit img_size=(256,320)', timm.create_model('maxvit_tiny_tf_224', img_size=(256,320), **C), [(256,320),(224,224)])
run('coatnet img_size=(256,320)', timm.create_model('coatnet_0_rw_224', img_size=(256,320), **C), [(256,320)])
run('hiera img_size=(256,320)', timm.create_model('hiera_tiny_224', img_size=(256,320), **C), [(256,320)])
run('levit img_size=(256,320)', timm.create_model('levit_128s', img_size=(256,320), **C), [(256,320)])
# in_chans
for n in ['resnet34','vit_small_patch16_224','swin_tiny_patch4_window7_224','efficientnet_b0']:
    m = timm.create_model(n, in_chans=13, **C); out = m(torch.randn(1,13,224,224)); print('in_chans13', n, 'ok', len(out))
from timm.models._manipulate import adapt_input_conv
w = torch.arange(3.).view(1,3,1,1)+1  # [1,2,3]
for c in [1,2,4,13]: print('adapt', c, adapt_input_conv(c, w).flatten().tolist())
# FeatureGetterNet norm / prune
m = timm.create_model('vit_small_patch16_224', out_indices=(2,5,8,11), **C); print('vit out_indices', m.feature_info.module_name(), len(m.model.blocks), m.model.norm)
m = timm.create_model('convnext_tiny', feature_cls='getter', out_indices=(1,3), pretrained=False, features_only=True); print('convnext getter info', m.feature_info.reduction())
m = timm.create_model('resnet34', feature_cls='hook', pretrained=False, features_only=True); print('hook', type(m).__name__)
m = timm.create_model('efficientnet_b0', feature_cls='hook', pretrained=False, features_only=True); print('eff hook', type(m).__name__)
