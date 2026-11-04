// Configurations for the model builder check (test_model_builder.py). Loads the page's own
// data.js + core.js, builds each configuration's state the way the form would, and prints, per
// case, the page's errors, the generated Python and the output shapes the page predicts.
//   node model_builder_cases.js <docs/builder dir>          ->  JSON on stdout
//   node model_builder_cases.js <docs/builder dir> groups   ->  the page's head sets, as JSON
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const dir = process.argv[2];
const ctx = { module: { exports: {} }, console };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(dir, "data.js"), "utf8") + "\n" +
  fs.readFileSync(path.join(dir, "core.js"), "utf8"), ctx);
const C = ctx.module.exports;
if (process.argv[3] === "groups") { process.stdout.write(JSON.stringify(C.HEAD_GROUPS)); process.exit(0); }

// no downloads in tests: random init everywhere, small tiles, batch 1
const small = (st, size = [96, 128]) => {
  st.inputs.forEach((r) => { r.weights = "none"; });
  st.size = size; st.batch = 1;
  return C.syncRes(st);
};
const state = (partial) => C.makeState(JSON.parse(JSON.stringify(partial)));
const head = (h) => ({ type: "segmentation", channels: 1, scope: "pixel", scale: 1, uncertainty: false,
  decoder: "", input: "", share: "", ...h });
const ex = (k) => C.EXAMPLES[k].state;

const cases = [];   // [name, state, expectError]
const ok = (name, st) => cases.push([name, st, false]);
const bad = (name, st) => cases.push([name, st, true]);

// every example, as shipped
for (const k of Object.keys(C.EXAMPLES)) ok(`example-${k}`, small(state(ex(k)), k === "sentinel2" ? [240, 240] : [96, 128]));

// fusion types x ops (two inputs; the Siamese ops included)
for (const ft of ["early", "intermediate", "late"]) {
  for (const op of ["concat", "add", "gated", "attention", "cross_attention", "diff", "concat_diff"]) {
    if (ft === "early" && op !== "concat") continue;
    const st = state({ ...ex("fusion"), fusion: { type: ft, op } });
    if (ft === "early") st.inputs.forEach((r) => { r.alignment = "same_grid"; });
    ok(`fusion-${ft}-${op}`, small(st));
  }
}
// relative pixel sizes, including non-integer and finer-than-primary ratios
for (const [rel, H] of [[6, 600], [6, 512], [0.5, 256], [3, 255]]) {
  const st = state(ex("fusion")); st.inputs[1].gsd = rel; ok(`gsd-${rel}-${H}`, small(st, [H, H]));
}
{ const st = state(ex("fusion")); st.inputs[1].backbone = "shared:optical"; ok("shared", small(st)); }
{ const st = state(ex("fusion")); st.inputs[1].backbone = "shared:optical"; st.inputs[1].share_norm = true; ok("shared-norm", small(st)); }
{ const st = state({ ...ex("fusion"), fusion: { type: "late", op: "concat" } }); st.inputs[1].backbone = "shared:optical"; ok("shared-late", small(st)); }

// decoders x output stride x width
for (const dec of ["unet", "unetplusplus", "fastai_unet", "fpn", "segformer", "deeplabv3plus", "upernet"]) {
  for (const os of [1, 2, 4]) {
    if (!["unet", "unetplusplus", "fastai_unet"].includes(dec) && os > 1) continue;
    for (const width of dec === "unet" ? ["full", "half", "quarter"] : ["full"]) {
      const st = state({ ...ex("multitask"), decoder: { type: dec, out_stride: os, width, input_cross: dec === "unet" && os === 1 && width === "half" } });
      st.heads[0].scale = os === 4 ? 0.25 : 1; st.inputs[0].backbone = "resnet18";
      ok(`dec-${dec}-${os}-${width}`, small(st));
    }
  }
}

// metadata
const metaBase = () => state(ex("metadata"));
{ const st = metaBase(); st.meta.fields.push({ name: "latlon", type: "coords" }, { name: "sensor", type: "categorical", num: 3 }); st.meta.mode = "head"; st.heads.push(head({ name: "cc", type: "regression", scope: "image" })); ok("meta-head", small(st)); }
{ const st = metaBase(); st.meta.mode = "head"; ok("meta-head-pixel-only", small(st)); }
{ const st = metaBase(); st.meta.mode = "channels"; ok("meta-channels", small(st)); }
{ const st = state({ ...ex("fusion"), meta: ex("metadata").meta }); ok("fusion-meta", small(st)); }

// backbones with special handling
{ const st = state(ex("segmentation")); st.inputs[0].backbone = "maxvit_tiny_tf_224"; ok("fixed-size", small(st, [512, 512])); }
{ const st = state(ex("segmentation")); st.inputs[0].backbone = "vit_small_patch16_224"; ok("vit", small(st)); }
{ const st = state({ ...ex("segmentation"), decoder: { type: "fpn" } }); st.inputs[0].backbone = "levit_128s"; ok("levit-fpn", small(st, [512, 512])); }
{ const st = state(ex("multitask")); st.inputs[0].name = "_x"; ok("single-input-name-unused", small(st)); }
// a fixed-size encoder on an input finer than the primary: the example must stay within its size
{ const st = state({ inputs: [{ name: "a", channels: 3, backbone: "resnet18" }, { name: "b", channels: 3, backbone: "maxvit_tiny_tf_224", alignment: "same_extent", gsd: 0.5 }], heads: [{ name: "m", channels: 2 }] });
  ok("fixed-size-finer-input", small(st, [512, 512])); }

// late fusion with per-head decoders and an image head
{ const st = state({ ...ex("fusion"), fusion: { type: "late", op: "gated" } });
  st.heads.push(head({ name: "edge", decoder: "unet" }), head({ name: "dist", type: "regression", scale: 0.5, decoder: "fpn" }));
  ok("late-task-image", small(st)); }

// time series: every aggregation, alone, with late fusion and with metadata
for (const agg of ["mean", "max", "attention", "concat", "diff"]) {
  const st = state(ex("timeseries")); st.inputs[0].temporal = agg; st.inputs[0].dates = agg === "diff" ? 2 : 3;
  ok(`ts-${agg}`, small(st));
}
{ const st = state({ inputs: [{ name: "s2", channels: 4, backbone: "resnet18", kind: "timeseries", dates: 4 }], heads: [{ name: "mask", channels: 3 }] }); ok("ts-alone", small(st)); }
{ const st = state({ ...ex("timeseries"), fusion: { type: "late", op: "concat" } }); ok("ts-late", small(st)); }
{ const st = state({ ...ex("timeseries"), meta: ex("metadata").meta }); ok("ts-meta", small(st)); }

// change detection: per-input heads with the model's decoder options or their own decoder
{ const st = state(ex("semantic")); st.decoder.out_stride = 2; ok("semantic-os2", small(st)); }
{ const st = state(ex("semantic")); st.heads[1].decoder = "fpn"; ok("semantic-own-fpn", small(st)); }
{ const st = state(ex("semantic")); st.heads[2].share = ""; ok("semantic-no-share", small(st)); }
{ const st = state(ex("semantic")); st.heads = st.heads.filter((h) => h.name === "lc1"); ok("one-per-input-head-returns-dict", small(st)); }

// pixel-head options: hidden layers (dilation, norm), output activations, bias init
{ const st = state(ex("multitask")); Object.assign(st.heads[0], { hidden: "16, 8", dilation: "2" });
  Object.assign(st.heads[1], { hidden: "8 4 8", dilation: "1, 4, 8", norm: "none" }); ok("head-hidden", small(st)); }
{ const st = state(ex("multitask")); Object.assign(st.heads[0], { hidden: "12", norm: "gn", scale: 0.5, bias_init: -2 }); ok("head-hidden-gn-scale", small(st)); }
{ const st = state(ex("segmentation")); st.heads[0].hidden = "8"; ok("head-hidden-single-returns-dict", small(st)); }
{ const st = state(ex("trees")); st.heads[2].activation = "softplus"; st.heads[2].bias_init = -4; ok("head-softplus", small(st)); }
for (const g of Object.keys(C.HEAD_GROUPS)) { const st = state({ ...ex("segmentation"), heads: [{ name: "landcover", type: "segmentation", channels: 3 }, ...C.HEAD_GROUPS[g]] }); ok(`group-${g}`, small(st)); }
{ const st = metaBase(); st.meta.mode = "head"; st.heads.push(C.normHead({ name: "d", type: "regression", activation: "softplus", hidden: [8] })); ok("meta-head-softplus", small(st)); }
{ const st = state({ ...ex("fusion"), fusion: { type: "late", op: "concat" } }); st.heads[0].hidden = "8, 8"; st.heads[0].dilation = "2, 3"; ok("late-head-hidden", small(st)); }
{ const st = state(ex("semantic")); for (const h of st.heads.slice(1)) Object.assign(h, { hidden: "16", norm: "gn" }); ok("semantic-share-hidden", small(st)); }
{ const st = state(ex("semantic")); st.heads[1].hidden = "16"; bad("err-share-hidden-differs", small(st)); }
{ const st = state(ex("multitask")); st.heads[0].hidden = "8, 0"; bad("err-hidden-zero", small(st)); }
{ const st = state(ex("multitask")); st.heads[0].hidden = "8, x"; bad("err-hidden-text", small(st)); }
{ const st = state(ex("multitask")); st.heads[0].hidden = "8, 8"; st.heads[0].dilation = "2, 2, 2"; bad("err-dilation-count", small(st)); }
{ const st = state(ex("multitask")); st.heads[0].hidden = "8"; st.heads[0].dilation = "0"; bad("err-dilation-zero", small(st)); }
{ const st = state(ex("multitask")); st.heads[1].activation = "softplus"; bad("err-softplus-uncertainty", small(st)); }
{ const st = state(ex("multitask")); st.heads[0].activation = "softplus"; bad("err-activation-segmentation", small(st)); }
{ const st = state(ex("trees")); st.heads[2].activation = "relu"; bad("err-relu-removed", small(st)); }
{ const st = state(ex("fusion")); st.heads[1].hidden = "8"; bad("err-image-head-hidden", small(st)); }
{ const st = state(ex("fusion")); st.heads[1].activation = "softplus"; bad("err-image-head-activation", small(st)); }
{ const st = state(ex("fusion")); st.heads[1].bias_init = 1; bad("err-image-head-bias", small(st)); }

// configurations the page must refuse (and ManySeg must reject too)
// levels: only levels a decoder reads are fused (DeepLabV3+ reads /4 and /32)
{ const st = state({ ...ex("fusion"), decoder: { type: "deeplabv3plus" } }); ok("fusion-deeplab-reads-4-32", small(st)); }
{ const st = state({ ...ex("fusion"), decoder: { type: "fpn" } }); ok("fusion-fpn-skips-2", small(st)); }
{ const st = state({ ...ex("classifier") }); st.inputs.push({ name: "sar", channels: 2, backbone: "resnet18", alignment: "same_grid", gsd: 1, res: 1, kind: "raster", temporal: "mean", dates: 6, weights: "none", share_norm: false });
  st.heads = st.heads.filter((h) => h.scope === "image"); ok("intermediate-image-heads-only", small(st)); }

{ const st = metaBase(); st.meta.fields[0].name = "train"; bad("err-field-reserved", small(st)); }
{ const st = state(ex("segmentation")); st.heads.push(head({ name: "x", channels: 2, scope: "image" })); bad("err-segmentation-per-image", small(st)); }
{ const st = state(ex("segmentation")); st.heads.push(head({ name: "x", type: "classification", channels: 2 })); bad("err-classification-per-pixel", small(st)); }
{ const st = state({ ...ex("fusion"), fusion: { type: "late", op: "concat" } }); st.heads = st.heads.filter((h) => h.scope === "image"); bad("err-late-unread-input", small(st)); }
{ const st = state({ ...ex("metadata"), decoder: { type: "fastai_unet" } }); ok("fastai-with-metadata", small(st)); }
{ const st = state({ ...ex("metadata"), decoder: { type: "unet", input_cross: true } }); ok("input-cross-with-metadata", small(st)); }
{ const st = state({ inputs: [{ name: "s2", channels: 4, backbone: "resnet18", kind: "timeseries", dates: 3 }], decoder: { type: "fastai_unet" }, heads: [{ name: "m", channels: 2 }] }); bad("err-fastai-with-timeseries", small(st)); }
{ const st = state({ inputs: [{ name: "a", channels: 3, backbone: "pvt_v2_b0.in1k" }, { name: "b", channels: 3, backbone: "levit_128s" }], heads: [{ name: "m", channels: 2 }] }); bad("err-trailing-levels", small(st, [224, 224])); }
{ const st = metaBase(); st.meta.fields[1].name = "doy"; bad("err-field-dup", small(st)); }
{ const st = metaBase(); st.meta.fields[1].std = 0; bad("err-std0", small(st)); }
{ const st = metaBase(); st.meta.fields[0].period = -1; bad("err-period", small(st)); }
{ const st = metaBase(); st.meta.fields.push({ name: "cat", type: "categorical", num: 0 }); bad("err-cat0", small(st)); }
{ const st = state(ex("multitask")); st.heads[0].name = "to"; bad("err-head-to", small(st)); }
{ const st = state(ex("fusion")); st.inputs[1].name = "_x"; bad("err-input-underscore", small(st)); }
{ const st = state(ex("sentinel2")); st.fusion.op = "diff"; bad("err-diff-three-inputs", small(st, [240, 240])); }
{ const st = state({ ...ex("timeseries"), fusion: { type: "early" } }); st.inputs[1].alignment = "same_grid"; bad("err-ts-early", small(st)); }
{ const st = state(ex("timeseries")); st.inputs[1].alignment = "same_grid"; st.inputs[1].channels = 10; st.inputs[1].backbone = "shared:s2"; bad("err-ts-shared", small(st)); }
{ const st = state(ex("timeseries")); st.inputs[1].name = "s2_mask"; bad("err-ts-name", small(st)); }
{ const st = state({ ...ex("timeseries"), meta: { ...ex("metadata").meta, mode: "channels" } }); bad("err-ts-meta-channels", small(st)); }
{ const st = state({ ...ex("semantic"), fusion: { type: "late", op: "concat" } }); bad("err-input-head-late", small(st)); }
{ const st = state(ex("semantic")); st.inputs[1].alignment = "same_extent"; st.inputs[1].gsd = 2; bad("err-input-head-extent", small(st)); }
{ const st = state(ex("semantic")); st.heads[2].channels = 5; bad("err-share-config", small(st)); }
{ const st = state(ex("semantic")); st.inputs[1].backbone = "convnext_tiny"; bad("err-share-specs", small(st)); }
{ const st = state(ex("segmentation")); small(st); st.inputs[0].backbone = "resnet50"; st.inputs[0].weights = "ssl4eo_s2_moco"; bad("err-weights-bands", st); }

const out = cases.map(([name, st, expectError]) => {
  const D = C.derive(st);
  return { name, expectError, errors: D.issues.filter((i) => i.level === "error").map((i) => i.text),
    code: C.generateCode(st, D), shapes: C.outputShapes(st, D), estimate: C.estimate(st, D).total };
});
process.stdout.write(JSON.stringify(out));
