import timm, torch, json, sys, traceback, inspect
torch.set_grad_enabled(False)
MODELS = {
 'vit': ('vit_small_patch16_224', dict(dynamic_img_size=True)),
 'eva': ('eva02_tiny_patch14_224', dict(dynamic_img_size=True)),
 'swin': ('swin_tiny_patch4_window7_224', dict(strict_img_size=False)),
 'convnext': ('convnext_tiny', {}),
 'efficientnet': ('efficientnet_b0', {}),
 'mnv3': ('mobilenetv3_large_100', {}),
 'mnv4': ('mobilenetv4_conv_small', {}),
 'regnet': ('regnety_016', {}),
 'resnet': ('resnet34', {}),
 'edgenext': ('edgenext_small', {}),
 'fastvit': ('fastvit_t8', {}),
 'maxvit': ('maxvit_tiny_tf_224', {}),
 'coatnet': ('coatnet_0_rw_224', {}),
 'hiera': ('hiera_tiny_224', {}),
 'mambaout': ('mambaout_femto', {}),
 'levit': ('levit_128s', {}),
 'caformer': ('caformer_s18', {}),
 'pvt_v2': ('pvt_v2_b0', {}),
 'nextvit': ('nextvit_small', {}),
}
SIZES = [(224,224),(256,320),(257,301),(97,131)]
only = sys.argv[1:] or list(MODELS)
for k in only:
    name, extra = MODELS[k]
    r = {'key': k, 'name': name}
    try:
        base = timm.create_model(name, pretrained=False)
        r['has_fi'] = hasattr(base, 'forward_intermediates')
        r['has_prune'] = hasattr(base, 'prune_intermediate_layers')
        r['has_set_input_size'] = hasattr(base, 'set_input_size')
        r['output_fmt_attr'] = str(getattr(base, 'output_fmt', None))
        r['pcfg_first_conv'] = base.pretrained_cfg.get('first_conv')
        r['base_fi_len'] = len(base.feature_info) if hasattr(base, 'feature_info') else None
        del base
    except Exception as e:
        r['base_err'] = repr(e)[:200]
    for tag, kw in [('default', {}), ('dyn', extra), ('getter', dict(feature_cls='getter'))]:
        if tag=='dyn' and not extra: continue
        res = {}
        try:
            m = timm.create_model(name, pretrained=False, features_only=True, **kw).eval()
            res['cls'] = type(m).__name__
            fi = m.feature_info
            res['ch'] = fi.channels(); res['red'] = fi.reduction(); res['mod'] = fi.module_name()
            res['out_fmt'] = str(getattr(m, 'output_fmt', None))
            for s in SIZES:
                try:
                    out = m(torch.randn(1,3,*s))
                    if isinstance(out, dict): out = list(out.values())
                    res[str(s)] = [tuple(o.shape) for o in out]
                except Exception as e:
                    res[str(s)] = 'FAIL: ' + (str(e).split('\n')[0])[:160]
        except Exception as e:
            res['create_err'] = (repr(e).split('\n')[0])[:200]
        r[tag] = res
    print(json.dumps(r), flush=True)
