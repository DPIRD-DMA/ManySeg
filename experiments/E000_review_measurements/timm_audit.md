# timm 1.0.30 encoder audit (CPU, pretrained=False)

Scripts: probe.py, probe2.py, probe3.py in this directory (probe.py's raw probe.jsonl was not kept).

## features_only per family (default feature_cls)

Shapes are C,H,W (NCHW) unless marked NHWC (H,W,C). Last stage only shown for brevity; full in probe.jsonl.

| key | model | wrapper | out fmt | channels | reductions | 224x224 | 256x320 | 257x301 | 97x131 | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| vit | vit_small_patch16_224 | FeatureGetterNet | NCHW | 384x3 (last 3 blocks) | 16,16,16 | 14x14 | FAIL strict size | FAIL | FAIL | dynamic_img_size=True: 256x320 OK 16x20, odd FAIL (div 16). +dynamic_img_pad=True: all OK, ceil (17x19, 7x9) |
| eva | eva02_tiny_patch14_224 | FeatureGetterNet | NCHW | 192x3 | 14,14,14 | 16x16 | FAIL | FAIL | FAIL | dyn: needs div 14 (256 fails). +pad: all OK, ceil (19x23, 19x22, 7x10) |
| swin | swin_tiny_patch4_window7_224 | FeatureListNet | **NHWC** | 96,192,384,768 | 4,8,16,32 | 7x7 | FAIL strict | FAIL | FAIL | strict_img_size=False: 256x320 OK (8x10), odd FAIL (div 4 at patch embed; dynamic_img_pad silently ignored). feature_cls='getter' gives NCHW. img_size=(256,320) or set_input_size() works for a fixed size |
| convnext | convnext_tiny | FeatureListNet | NCHW | 96,192,384,768 | 4..32 | 7x7 | 8x10 | 8x9 | 3x4 | floor (patchify stem k4s4) |
| efficientnet | efficientnet_b0 | EfficientNetFeatures | NCHW | 16,24,40,112,320 | 2..32 | 7x7 | 8x10 | 9x10 | 4x5 | ceil |
| mnv3 | mobilenetv3_large_100 | MobileNetV3Features | NCHW | 16,24,40,112,960 | 2..32 | 7x7 | 8x10 | 9x10 | 4x5 | ceil |
| mnv4 | mobilenetv4_conv_small | MobileNetV3Features | NCHW | 32,32,64,96,960 | 2..32 | 7x7 | 8x10 | 9x10 | 4x5 | ceil |
| regnet | regnety_016 | FeatureListNet | NCHW | 32,48,120,336,888 | 2..32 | 7x7 | 8x10 | 9x10 | 4x5 | ceil |
| resnet | resnet34 | FeatureListNet | NCHW | 64,64,128,256,512 | 2..32 | 7x7 | 8x10 | 9x10 | 4x5 | ceil |
| edgenext | edgenext_small | FeatureListNet | NCHW | 48,96,160,304 | 4..32 | 7x7 | 8x10 | 8x9 | 3x4 | floor |
| fastvit | fastvit_t8 | FeatureListNet | NCHW | 48,96,192,384 | 4..32 | 7x7 | 8x10 | 9x10 | 4x5 | ceil |
| maxvit | maxvit_tiny_tf_224 | FeatureListNet | NCHW | 64,64,128,256,512 | 2..32 | 7x7 | FAIL window 7 | FAIL | FAIL | fixed size; img_size=(256,320) at create works, then only that size |
| coatnet | coatnet_0_rw_224 | FeatureListNet | NCHW | 64,96,192,384,768 | 2..32 | 7x7 | FAIL rel-pos 196 | FAIL | FAIL | fixed size; img_size= at create works |
| hiera | hiera_tiny_224 | FeatureGetterNet | NCHW | 96,192,384,768 | 4..32 | 7x7 | FAIL pos-embed | FAIL | FAIL | fixed size; img_size= at create works |
| mambaout | mambaout_femto | FeatureListNet | **NHWC** | 48,96,192,288 | 4..32 | 7x7 | 8x10 | 9x10 | 4x5 | ceil; getter -> NCHW |
| levit | levit_128s | FeatureGetterNet | NCHW | 128,256,384 | 16,32,64 | 4x4 | FAIL attn bias | FAIL | FAIL | fixed size; img_size= at create works |
| caformer | caformer_s18 | FeatureListNet | NCHW | 64,128,320,512 | 4..32 | 7x7 | 8x10 | 8x10 | 3x5 | mixed (stem k7s4p2 floor-ish, then ceil) |
| pvt_v2 | pvt_v2_b0 | FeatureListNet | NCHW | 32,64,160,256 | 4..32 | 7x7 | 8x10 | 9x10 | 4x5 | ceil |
| nextvit | nextvit_small | FeatureListNet | NCHW | 96,256,512,1024 | 4..32 | 7x7 | 8x10 | 9x10 | 4x5 | ceil |

All 19 have forward_intermediates + prune_intermediate_layers; feature_cls='getter' works for all and always returns NCHW with identical shapes.
Only vit, eva, swin have set_input_size().

## in_chans
create_model(in_chans=13) OK for resnet34, vit, swin, effnet_b0 (random init).
adapt_input_conv on weight [1,2,3] (per out-chan RGB): 1 -> [6] (sum); 2 -> [1.5,3.0]; 4 -> [0.75,1.5,2.25,0.75]; 13 -> RGB tiled, scaled by 3/13.

## torch.export / torch.compile (dynamic H,W)

export_free: Dim('H',64..2048), Dim('W',64..2048); export_m32: H=32*Dim, W=32*Dim. Second call at the second size (257x301 for CNNs and vitpad, 224x288 for vit and swin).
compile: CUDA, fullgraph=True, dynamic=True, two sizes, shapes compared against eager.

| model | export_free | export_m32 | compile fullgraph dynamic |
|---|---|---|---|
| resnet34 features_only | OK | OK | OK (23s) |
| convnext_tiny features_only | OK | OK | OK (20s) |
| efficientnet_b0 features_only | OK | OK | OK (31s) |
| vit_s16 dynamic_img_size (getter) | FAIL: guard H%16==0, suggests 16*Dim | OK | OK (16s) |
| vit_s16 dynamic_img_size+dynamic_img_pad | OK | OK | OK (19s) |
| swin_tiny strict_img_size=False | TIMEOUT (>600s, killed) | TIMEOUT (>600s, killed) | OK (79s; output NHWC) |

Note: an exported program built with 32*Dim still ran on a 257x301 input through ep.module() (resnet, vitpad) without a runtime divisibility check.
Static-shape swin export (224x224, no dynamic_shapes): OK in 2s. The dynamic-shape hang is likely symbolic reasoning over the dynamic_mask path (get_attn_mask, swin_transformer.py:350, built per forward when strict_img_size=False).
