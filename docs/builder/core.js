// ManySeg model builder: pure logic (no DOM). State -> create_model code, shapes, estimates.
// Loaded after data.js (global DATA). Also runs under node for the code-validity check.

const PY_KEYWORDS = new Set(("False None True and as assert async await break class continue def del " +
  "elif else except finally for from global if import in is lambda nonlocal not or pass raise " +
  "return try while with yield").split(" "));
const FIXED_DECODERS = ["fpn", "segformer", "deeplabv3plus", "upernet"];
const UNET_FAMILY = ["unet", "unetplusplus", "fastai_unet"];
const DEFAULT_UNET = [256, 128, 64, 32, 16];
const FIXED_NATIVE = 224;  // every fixed-size backbone in the compatibility table runs at 224

function backboneInfo(name) {
  return DATA.backbones.find((b) => b.name === name) || DATA.backbones[0];
}

// remote-sensing weights tags (manyseg.weights) that exist for this timm architecture
function weightTags(backbone) {
  const arch = String(backbone).split(".")[0];
  return Object.values(DATA.weights || {}).filter((w) => w.model === arch);
}

// ---------- examples ----------

const EXAMPLES = {
  segmentation: {
    label: "Segmentation",
    blurb: "One 4-channel image, 3 classes. The library default: ResNet-34 U-Net.",
    state: {
      inputs: [{ name: "image", channels: 4, backbone: "resnet34" }],
      heads: [{ name: "mask", type: "segmentation", channels: 3 }],
    },
  },
  fast: {
    label: "Fast preset",
    blurb: "The measured 'balanced' preset: PVTv2-b0 with a half-width U-Net, about 2x faster.",
    state: {
      inputs: [{ name: "image", channels: 4, backbone: "pvt_v2_b0.in1k" }],
      decoder: { type: "unet", width: "half" },
      heads: [{ name: "mask", type: "segmentation", channels: 2 }],
    },
  },
  fusion: {
    label: "Optical + SAR",
    blurb: "Two sensors with their own encoders; the radar is at half resolution over the same extent.",
    state: {
      inputs: [
        { name: "optical", channels: 4, backbone: "convnext_tiny" },
        { name: "sar", channels: 2, backbone: "resnet18", alignment: "same_extent", gsd: 2 },
      ],
      fusion: { type: "intermediate", op: "gated" },
      heads: [
        { name: "water", type: "segmentation", channels: 2 },
        { name: "water_area", type: "regression", channels: 1, scope: "image" },
      ],
    },
  },
  sentinel2: {
    label: "Sentinel-2 10/20/60 m",
    blurb: "Sentinel-2 band groups at their native pixel sizes, fused over the same extent: no resampling in the data pipeline.",
    state: {
      inputs: [
        { name: "s2_10m", channels: 4, backbone: "resnet34" },
        { name: "s2_20m", channels: 6, backbone: "resnet18", alignment: "same_extent", gsd: 2 },
        { name: "s2_60m", channels: 3, backbone: "resnet18", alignment: "same_extent", gsd: 6 },
      ],
      fusion: { type: "intermediate", op: "concat" },
      heads: [{ name: "landcover", type: "segmentation", channels: 10 }],
      size: [600, 600],
    },
  },
  change: {
    label: "Change detection",
    blurb: "Before and after images through one Siamese encoder, fused with concat_diff. Siamese diff and concat_diff beat stacking both dates on two datasets (E022).",
    state: {
      inputs: [
        { name: "before", channels: 3, backbone: "resnet34" },
        { name: "after", channels: 3, backbone: "shared:before", share_norm: true },
      ],
      fusion: { type: "intermediate", op: "concat_diff" },
      heads: [{ name: "change", type: "segmentation", channels: 2 }],
    },
  },
  semantic: {
    label: "Semantic change",
    blurb: "What changed into what: a change head on the fused features plus a land-cover head per date, sharing weights across dates. Shared per-input heads beat land cover from the fused features (E023).",
    state: {
      inputs: [
        { name: "t1", channels: 3, backbone: "resnet34" },
        { name: "t2", channels: 3, backbone: "shared:t1", share_norm: true },
      ],
      fusion: { type: "intermediate", op: "concat_diff" },
      heads: [
        { name: "change", type: "segmentation", channels: 2 },
        { name: "lc1", type: "segmentation", channels: 7, input: "t1" },
        { name: "lc2", type: "segmentation", channels: 7, input: "t2", share: "lc1" },
      ],
    },
  },
  timeseries: {
    label: "Time series",
    blurb: "Six Sentinel-2 dates through one encoder, aggregated per level with temporal attention, plus a DEM. Several dates beat one by a wide margin (E021).",
    state: {
      inputs: [
        { name: "s2", channels: 10, backbone: "resnet18", kind: "timeseries", temporal: "attention", dates: 6 },
        { name: "dem", channels: 1, backbone: "resnet18", alignment: "same_extent", gsd: 2 },
      ],
      heads: [{ name: "crop", type: "segmentation", channels: 20 }],
      size: [256, 256],
    },
  },
  multitask: {
    label: "Multi-task",
    blurb: "Land cover, per-pixel height and a scene-level flag from one encoder.",
    state: {
      inputs: [{ name: "image", channels: 4, backbone: "resnet50" }],
      heads: [
        { name: "landcover", type: "segmentation", channels: 8 },
        { name: "height", type: "regression", channels: 1, uncertainty: true },
        { name: "flooded", type: "classification", channels: 2, scope: "image" },
      ],
    },
  },
  classifier: {
    label: "Classifier",
    blurb: "Multi-label scene classifier with an auxiliary segmentation head (E020: +2.3 mAP).",
    state: {
      inputs: [{ name: "s2", channels: 12, backbone: "resnet50" }],
      heads: [
        { name: "labels", type: "classification", channels: 19, scope: "image" },
        { name: "aux_mask", type: "segmentation", channels: 19 },
      ],
    },
  },
  fields: {
    label: "Field boundaries",
    blurb: "Instance heads (extent, boundary, distance), each with its own U-Net decoder (E010).",
    state: {
      inputs: [{ name: "image", channels: 4, backbone: "convnext_tiny" }],
      heads: [
        { name: "extent", type: "segmentation", channels: 1, decoder: "unet" },
        { name: "boundary", type: "segmentation", channels: 1, decoder: "unet" },
        { name: "distance", type: "regression", channels: 1, decoder: "unet" },
      ],
    },
  },
  trees: {
    label: "Tree counting",
    blurb: "Centre heatmap, offsets and a density map for counting (E011).",
    state: {
      inputs: [{ name: "image", channels: 3, backbone: "resnet34" }],
      heads: [
        { name: "heatmap", type: "regression", channels: 1 },
        { name: "offset", type: "regression", channels: 2 },
        { name: "density", type: "regression", channels: 1 },
      ],
    },
  },
  metadata: {
    label: "With metadata",
    blurb: "Sentinel-2 plus acquisition date and sun elevation, injected with FiLM.",
    state: {
      inputs: [{ name: "s2", channels: 13, backbone: "resnet34" }],
      meta: {
        name: "meta", mode: "film",
        fields: [
          { name: "doy", type: "cyclic", period: 365.25 },
          { name: "sun_elev", type: "continuous", mean: 45, std: 15 },
        ],
      },
      heads: [{ name: "cloud", type: "segmentation", channels: 4 }],
    },
  },
};

const HEAD_GROUPS = {
  instances: [
    { name: "extent", type: "segmentation", channels: 1 },
    { name: "boundary", type: "segmentation", channels: 1 },
    { name: "distance", type: "regression", channels: 1 },
  ],
  points: [
    { name: "heatmap", type: "regression", channels: 1 },
    { name: "offset", type: "regression", channels: 2 },
  ],
  density: [{ name: "density", type: "regression", channels: 1 }],
};

function normInput(i, idx) {
  return {
    name: i.name ?? (idx === 0 ? "image" : `input${idx + 1}`),
    channels: i.channels ?? 3,
    backbone: i.backbone ?? "resnet34",
    alignment: idx === 0 ? "same_grid" : (i.alignment ?? "same_grid"),
    gsd: idx === 0 ? 1 : (i.gsd ?? 1),   // pixel size relative to the primary (6 = one pixel spans 6 primary pixels)
    res: 1,
    kind: i.kind ?? "raster",            // "raster" (B, C, H, W) or "timeseries" (B, T, C, H, W)
    temporal: i.temporal ?? "mean",      // time-series aggregation per feature level
    dates: i.dates ?? 6,                 // T in the example (fixed for "concat", 2 for "diff")
    weights: i.weights ?? "imagenet",    // "imagenet", "none" or a manyseg.weights tag
    share_norm: !!i.share_norm,          // with a "shared:" backbone: share BatchNorm too
  };
}

// res = primary pixel size / this input's pixel size (1/6 for 60 m next to 10 m; 2 for a finer input)
function syncRes(st) {
  st.inputs[0].gsd = 1;
  const g0 = 1;
  st.inputs.forEach((r, i) => {
    if (i === 0 || r.alignment === "same_grid") { r.res = 1; if (i > 0) r.gsd = g0; }
    else r.res = g0 / r.gsd;
  });
  return st;
}

// "1/6 res", "2x res": resolution relative to the primary input
function resLabel(res) {
  const f = (v) => String(Math.round(v * 100) / 100);
  if (Math.abs(res - 1) < 1e-9) return "same res";
  return res < 1 ? `1/${f(1 / res)} res` : `${f(res)}\u00d7 res`;
}

function inputPixels(st, r, H) {
  return Math.max(1, Math.round(H * st.inputs[0].gsd / r.gsd));
}

function normHead(h) {
  return {
    name: h.name, type: h.type ?? "segmentation", channels: h.channels ?? 1,
    scope: h.scope ?? "pixel", scale: h.scale ?? 1, uncertainty: !!h.uncertainty,
    decoder: h.decoder ?? "",
    input: h.input ?? "",                // per-input head: read this raster input's own features
    share: h.share ?? "",                // reuse another per-input head's decoder and head
  };
}

function makeState(partial) {
  const p = JSON.parse(JSON.stringify(partial));
  const inputs = p.inputs.map(normInput);
  if (p.pretrained === false) inputs.forEach((r) => { r.weights = "none"; });  // older saved states
  return syncRes({
    channel_adapter: p.channel_adapter ?? "repeat",
    inputs,
    meta: p.meta ?? null,
    fusion: { type: "intermediate", op: "concat", ...(p.fusion || {}) },
    decoder: { type: "unet", out_stride: 1, width: "full", input_cross: false, ...(p.decoder || {}) },
    heads: p.heads.map(normHead),
    size: p.size ?? [512, 512],
    batch: p.batch ?? 2,
  });
}

// ---------- derived structure ----------

function derive(s) {
  const rasters = s.inputs;
  const multi = rasters.length > 1;
  const ftype = multi ? s.fusion.type : null;
  const pixelHeads = s.heads.filter((h) => h.scope === "pixel");
  const imageHeads = s.heads.filter((h) => h.scope === "image");
  const needsShared = pixelHeads.some((h) => !h.decoder && !h.input);
  const taskDecoders = pixelHeads.filter((h) => h.decoder || h.input).map((h) => h.name);
  const series = rasters.filter((r) => r.kind === "timeseries");
  const useInputs = multi || !!s.meta || series.length > 0;   // otherwise in_channels=
  const rawImageBlocked = multi || series.length > 0;          // fastai_unet / input_cross
  const issues = [];   // {level: "error"|"warn"|"tip", text}

  // encoders: early fusion -> one encoder on stacked channels; shared -> reuse
  const primary = backboneInfo(ftype === "early" ? rasters[0].backbone : rasters[0].backbone);
  const encoders = [];
  if (ftype === "early") {
    encoders.push({ input: rasters.map((r) => r.name).join(" + "), info: primary,
      channels: rasters.reduce((a, r) => a + r.channels, 0), shared: null, inputs: rasters });
  } else {
    for (const r of rasters) {
      const isShared = r.backbone.startsWith("shared:");
      const info = backboneInfo(isShared ? rasters[0].backbone : r.backbone);
      encoders.push({ input: r.name, info, channels: r.channels,
        shared: isShared ? r.backbone.slice(7) : null, inputs: [r] });
    }
  }


  // decoder output specs
  function decKey(type, os, width) {
    if (FIXED_DECODERS.includes(type)) return type;
    if (type === "unet") return `unet/${os}/${width}`;
    return `${type}/${os}`;
  }
  function decSpec(info, type, os, width) {
    const v = info.dec[decKey(type, os, width)];
    if (typeof v === "string") return { error: v };
    return { channels: v[0], stride: v[1], params_m: v[2], read: v[3] || [] };
  }
  const d = s.decoder;
  let shared = null;
  if (needsShared) {
    shared = { type: d.type, ...decSpec(primary, d.type, d.out_stride, d.width) };
    if (shared.error) issues.push({ level: "error", text: `${d.type} on ${primary.name}: ${shared.error}` });
    if (d.type === "unet" && d.input_cross) shared.params_m += 0.007;
    // decoders that read the raw image need one plain image (metadata alongside is fine)
    if (d.type === "fastai_unet" && rawImageBlocked)
      issues.push({ level: "error", text: "fastai_unet reads the raw input image, which models with several images or a time series don't have. Use a decoder that reads features only, e.g. unet." });
    if (d.type === "unet" && d.input_cross && rawImageBlocked)
      issues.push({ level: "error", text: "input_cross reads the raw input image, which models with several images or a time series don't have." });
  }
  const task = {};
  for (const h of pixelHeads.filter((h) => h.decoder && !h.input)) {
    task[h.name] = { type: h.decoder, ...decSpec(primary, h.decoder, 1, "full") };
    if (task[h.name].error) issues.push({ level: "error", text: `${h.decoder} on ${primary.name}: ${task[h.name].error}` });
    if (h.decoder === "fastai_unet" && rawImageBlocked)
      issues.push({ level: "error", text: `head '${h.name}': fastai_unet reads the raw input image, which models with several images or a time series don't have.` });
  }
  // per-input heads ("input": <raster>): their own decoder on that input's unfused features,
  // by default the model's decoder type and options; "share" reuses another one's decoder + head
  const encOf = (name) => encoders.find((e) => e.input === name);
  for (const h of s.heads.filter((h) => h.input)) {
    const r = rasters.find((x) => x.name === h.input);
    const err = (t) => issues.push({ level: "error", text: `head '${h.name}': ${t}` });
    if (!multi) { err("reading one input needs several raster inputs."); continue; }
    if (ftype !== "intermediate") { err(`per-input heads need intermediate fusion (${ftype} fusion ${ftype === "late" ? "already decodes each input" : "has one encoder"}).`); continue; }
    if (!r) { err(`there is no raster input '${h.input}'.`); continue; }
    if (h.scope !== "pixel") { err("per-input heads are pixel heads."); continue; }
    if (r.kind === "timeseries") { err("a per-input head cannot read a time series yet."); continue; }
    if (r.alignment !== "same_grid") { err(`per-input heads need an input on the primary grid; '${r.name}' is same extent.`); continue; }
    const info = encOf(r.name).info;
    const type = h.decoder || d.type;
    const spec = h.decoder ? decSpec(info, type, 1, "full") : decSpec(info, type, d.out_stride, d.width);
    if (type === "fastai_unet" || (!h.decoder && d.type === "unet" && d.input_cross))
      err("its decoder would read the raw input image (fastai_unet or input_cross), which per-input heads do not support.");
    task[h.name] = { type, ...spec, input: r.name, width: h.decoder ? "full" : d.width };
    if (spec.error) err(`${type} on ${info.name}: ${spec.error}`);
  }
  for (const h of s.heads.filter((h) => h.share)) {
    const o = s.heads.find((x) => x.name === h.share);
    const err = (t) => issues.push({ level: "error", text: `head '${h.name}': ${t}` });
    if (!h.input || !o || !o.input || o.share) { err(`'share' links two per-input heads, and '${h.share}' must be one that does not share itself.`); continue; }
    const cfg = (x) => [x.type, x.channels, x.scope, x.scale, x.uncertainty].join();
    if (cfg(h) !== cfg(o)) { err(`'share' needs the same type, channels, scale and uncertainty as '${o.name}'.`); continue; }
    const a = encOf(h.input), b = encOf(o.input);
    if (a && b && (a.info.strides.join() !== b.info.strides.join() || a.info.channels.join() !== b.info.channels.join())) { err(`'share' needs inputs with the same encoder levels ('${h.input}' and '${o.input}' differ).`); continue; }
    if (task[o.name]) task[h.name] = { ...task[o.name], input: h.input, sharesWith: o.name };
  }

  // which levels are read, as in DenseModel._restrict_to_read_levels: each decoder's read levels
  // (ManySeg's gradient probe, exported in data.js) plus the deepest when a head pools the
  // encoder. Only read levels are fused, and an input (or the deepest levels of one) that nothing
  // reads is an error in create_model, since its parameters would never get a gradient.
  let fused = [];
  const okSpec = (t) => t && !t.error;
  const deepestOf = (info) => info.strides[info.strides.length - 1];
  if (multi && ftype !== "early") {
    const need = new Map(encoders.map((e) => [e.input, new Set()]));
    const fusedDecs = [shared, ...Object.values(task).filter((t) => !t.input)].filter(okSpec);
    if (ftype === "intermediate") {
      const read = new Set(fusedDecs.flatMap((t) => t.read));
      if (imageHeads.length) read.add(deepestOf(primary));
      const others = new Set(encoders.slice(1).flatMap((e) => e.info.strides));
      fused = primary.strides.filter((st) => read.has(st) && others.has(st) &&
        (s.fusion.op !== "cross_attention" || st >= 16));
      for (const st of read) if (primary.strides.includes(st)) need.get(encoders[0].input).add(st);
      for (const e of encoders.slice(1)) for (const st of fused) if (e.info.strides.includes(st)) need.get(e.input).add(st);
      for (const t of Object.values(task).filter((t) => t.input && okSpec(t))) {
        const e = encOf(t.input);
        if (e) for (const st of t.read) need.get(e.input).add(st);
      }
    } else {   // late: every input runs every fused decoder on its own levels
      for (const e of encoders) for (const t of fusedDecs) {
        const v = e.info.dec[t === shared ? decKey(t.type, d.out_stride, d.width) : decKey(t.type, 1, "full")];
        if (Array.isArray(v)) for (const st of v[3] || []) need.get(e.input).add(st);
      }
      if (imageHeads.length) need.get(encoders[0].input).add(deepestOf(primary));
    }
    const allOk = !issues.some((i) => i.level === "error");
    for (const e of encoders) {
      const n = need.get(e.input);
      if (!n.size && !e.shared && allOk)
        issues.push({ level: "error", text: ftype === "late"
          ? `input '${e.input}' would never be read: late fusion combines inputs in the decoder, and no head reads a decoder. Add a pixel head, use intermediate fusion, or drop the input.`
          : `input '${e.input}' would never be read: no fused level that a decoder or head reads includes it. Add a head that reads a decoder, change the fusion, or drop the input.` });
      else if (n.size && Math.max(...n) < deepestOf(e.info) && allOk)
        issues.push({ level: "error", text: `input '${e.input}': its levels deeper than /${Math.max(...n)} are never read (nothing fuses or decodes them), so ManySeg refuses it. Use an encoder whose deepest level matches the primary's.` });
    }
  }

  // validation
  // the same rules as manyseg.registry.check_module_names, applied to inputs, heads and fields
  const names = new Set();
  const reserved = new Set(DATA.reserved_names || []);
  // a lone raster without metadata becomes in_channels=, so its name never reaches ManySeg
  const namedInputs = useInputs;
  for (const [kind, n] of [...(namedInputs ? rasters.map((r) => ["input", r.name]) : []),
    ...(s.meta ? [["input", s.meta.name]] : []), ...s.heads.map((h) => ["head", h.name]),
    ...(s.meta ? s.meta.fields.map((f) => ["field", f.name]) : [])]) {
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(n)) issues.push({ level: "error", text: `${kind} name '${n}' must start with a letter and use only letters, digits and underscores.` });
    else if (PY_KEYWORDS.has(n)) issues.push({ level: "error", text: `${kind} name '${n}' is a Python keyword; torch.export reads modules as attributes.` });
    else if (reserved.has(n)) issues.push({ level: "error", text: `${kind} name '${n}' is reserved: it is an attribute of torch's ModuleDict. Try '${n}_'.` });
    const key = `${kind}:${n}`;
    if (names.has(key)) issues.push({ level: "error", text: `Two ${kind}s are called '${n}'.` });
    names.add(key);
  }
  if (s.meta) {
    for (const f of s.meta.fields) {
      if (f.type === "continuous" && !(f.std > 0)) issues.push({ level: "error", text: `field '${f.name}': std must be above 0.` });
      if (f.type === "cyclic" && !(f.period > 0)) issues.push({ level: "error", text: `field '${f.name}': period must be above 0.` });
      if (f.type === "categorical" && !(Number.isInteger(f.num) && f.num >= 1)) issues.push({ level: "error", text: `field '${f.name}': needs at least 1 category.` });
      if (f.type === "continuous" && !Number.isFinite(f.mean ?? 0)) issues.push({ level: "error", text: `field '${f.name}': mean must be a number.` });
    }
    if (s.meta.mode === "head" && !imageHeads.length)
      issues.push({ level: "tip", text: "No image-level heads, so the metadata bias is added to the pixel heads ({\"mode\": \"head\", \"pixel\": True})." });
  }
  if (!s.heads.length) issues.push({ level: "error", text: "Add at least one head." });
  for (const r of rasters) if (!(r.channels >= 1)) issues.push({ level: "error", text: `input '${r.name}' needs at least 1 channel.` });
  for (const h of s.heads) if (!(h.channels >= 1)) issues.push({ level: "error", text: `head '${h.name}' needs at least 1 channel.` });
  for (const r of rasters) if (!(r.gsd > 0)) issues.push({ level: "error", text: `input '${r.name}' needs a relative pixel size above 0.` });
  {
    const [H0, W0] = s.size;
    for (const r of rasters.slice(1)) {
      if (r.alignment !== "same_extent" || !(r.gsd > 0)) continue;
      const px = Math.min(inputPixels(s, r, H0), inputPixels(s, r, W0));
      if (px < 64) issues.push({ level: "warn", text: `At ${H0}\u00d7${W0} px of ${s.inputs[0].name}, '${r.name}' (${resLabel(r.res)}) is only ${px} px across, so its /32 level is ${Math.max(1, Math.ceil(px / 32))} px${px <= 32 ? " (training at batch 1 then fails in BatchNorm)" : ""}. Use larger tiles for this input.` });
    }
  }
  if (ftype === "early" && rasters.some((r) => r.alignment !== "same_grid"))
    issues.push({ level: "error", text: "Early fusion stacks channels, so every input must be on the same grid." });

  // time series
  if (ftype === "early" && series.length)
    issues.push({ level: "error", text: "Early fusion cannot stack a time series (it has a date axis); use intermediate or late fusion." });
  for (const r of rasters) {
    if (!r.backbone.startsWith("shared:")) continue;
    const target = rasters.find((x) => x.name === r.backbone.slice(7));
    if (r.kind === "timeseries" || (target && target.kind === "timeseries"))
      issues.push({ level: "error", text: `input '${r.name}': a shared encoder cannot involve a time series yet; give each input its own encoder.` });
  }
  for (const r of series) {
    const taken = [...rasters.map((x) => x.name), ...(s.meta ? [s.meta.name] : [])];
    for (const suffix of ["mask", "dates"])
      if (taken.includes(`${r.name}_${suffix}`)) issues.push({ level: "error", text: `input name '${r.name}_${suffix}' is reserved for the ${suffix} of time series '${r.name}'.` });
    if (r.temporal === "concat" && !(Number.isInteger(r.dates) && r.dates >= 1))
      issues.push({ level: "error", text: `input '${r.name}': 'concat' needs a fixed number of dates (at least 1).` });
    if (r.temporal === "diff" && r.dates !== 2)
      issues.push({ level: "error", text: `input '${r.name}': 'diff' compares exactly 2 dates.` });
    if (!(Number.isInteger(r.dates) && r.dates >= 1))
      issues.push({ level: "error", text: `input '${r.name}': the example needs at least 1 date.` });
  }
  if (s.meta && s.meta.mode === "channels" && rasters[0].kind === "timeseries")
    issues.push({ level: "error", text: "Metadata as extra channels cannot target a time series yet; it targets the first raster input." });

  // Siamese change ops compare exactly two inputs
  if (multi && ftype !== "early" && ["diff", "concat_diff"].includes(s.fusion.op) && rasters.length !== 2)
    issues.push({ level: "error", text: `'${s.fusion.op}' fusion compares exactly two inputs (before / after); there are ${rasters.length}.` });

  // pretrained weights: a remote-sensing tag must exist for the architecture and match the bands
  const weighed = ftype === "early" ? [{ r: rasters[0], ch: encoders[0].channels }]
    : rasters.filter((r) => !r.backbone.startsWith("shared:")).map((r) => ({ r, ch: r.channels }));
  for (const { r, ch } of weighed) {
    if (r.weights === "imagenet" || r.weights === "none") continue;
    const w = (DATA.weights || {})[`${r.backbone.split(".")[0]}.${r.weights}`];
    if (!w) issues.push({ level: "error", text: `input '${r.name}': there are no '${r.weights}' weights for ${r.backbone}.` });
    else if (w.bands !== ch) issues.push({ level: "error", text: `input '${r.name}': '${r.weights}' weights expect ${w.bands} channels in a fixed band order, not ${ch}.` });
    else issues.push({ level: "tip", text: `'${r.weights}' weights expect ${w.units}. Normalise with the mean and std in manyseg.weights.WEIGHTS["${r.backbone.split(".")[0]}.${r.weights}"] (clipped to [0, 1] where clip is set); bands your data lacks can be filled with the pretraining mean.` });
  }
  for (const { r, ch } of weighed) {
    const tags = weightTags(r.backbone).filter((w) => w.bands === ch && w.tag.startsWith("ssl4eo"));
    if (r.weights === "imagenet" && tags.length)
      issues.push({ level: "tip", text: `input '${r.name}': ${tags.map((w) => w.tag).join(" and ")} pretraining exists for ${r.backbone} with ${ch} bands. On three Sentinel-2 datasets ssl4eo_s2_dino beat ImageNet weights with a ResNet-50 (E024); pick it under Weights.` });
  }

  // advice from the experiments
  for (const e of encoders) {
    const b = e.info;
    if (b.size === "fixed") issues.push({ level: "warn", text: `${b.name} only runs at one size (${FIXED_NATIVE} px): smaller inputs are padded, larger ones are an error. Pass backbone_kwargs={"img_size": ...} for another size.${(() => { const [h, w] = clampForFixed(s, encoders); return h !== s.size[0] || w !== s.size[1] ? ` The example uses ${h}\u00d7${w} px for ${rasters[0].name} instead of ${s.size[0]}\u00d7${s.size[1]}.` : ""; })()}` });
    if (b.export === "fail") issues.push({ level: "warn", text: `${b.name}: torch.export and ONNX need a fixed input size (window attention, a timm limitation). torch.compile and eager work at any size.` });
    if (b.vit) issues.push({ level: "tip", text: `${b.name} is a plain ViT: a neck turns its single-scale tokens into a /4 to /32 pyramid.` });
  }
  if (ftype === "intermediate" || ftype === "late")
    issues.push({ level: "tip", text: "Stacking all channels into one input (early fusion) was as good as fancier fusion on the datasets we tried. Start there and compare." });
  if (ftype === "intermediate" && s.fusion.op === "cross_attention")
    issues.push({ level: "tip", text: "cross_attention is quadratic in tokens, so it fuses strides 16 and 32 only." });
  if (s.meta) issues.push({ level: "tip", text: "Metadata showed no benefit on three datasets (E009, E013, E016). Compare against the same model without it, on splits that hold out whole regions or events." });
  if (needsShared && d.input_cross) issues.push({ level: "tip", text: "input_cross helped water mapping (E012) but not pets or clouds. It suits targets decided by a pixel's own spectrum." });
  if (!pixelHeads.length && imageHeads.length) issues.push({ level: "tip", text: "No pixel heads, so no decoder is built: this is a classifier. With pixel labels, an auxiliary segmentation head raised mAP by 2.3 in E020; drop it before deployment with model.select_heads([...])." });
  if (s.heads.some((h) => h.uncertainty)) issues.push({ level: "tip", text: "Uncertainty heads return [mean, log_variance] (2x channels), for a Gaussian NLL loss." });
  if (series.length) issues.push({ level: "tip", text: "Stacking all dates as channels of one input is a strong, cheaper baseline: it beat the time-series input on PASTIS crop types, while the time-series input with attention won on AI4Boundaries (E021). Try both." });
  if (multi && ftype !== "early" && ["diff", "concat_diff"].includes(s.fusion.op)) issues.push({ level: "tip", text: "Neither difference op won on both change datasets in E022: diff was best on Sentinel-2 (OSCD) and worst of the Siamese ops on 0.5 m imagery (LEVIR-CD). Try both." });
  if (multi && ftype === "intermediate" && rasters.length === 2 && s.fusion.op === "concat" && rasters[1].backbone === `shared:${rasters[0].name}`)
    issues.push({ level: "tip", text: "For change detection, Siamese diff or concat_diff beat stacking both dates on both datasets in E022; Siamese concat did on one (LEVIR-CD) only." });

  return { rasters, multi, ftype, primary, encoders, fused, shared, task, pixelHeads, imageHeads,
    needsShared, taskDecoders, series, useInputs, issues };
}

// ---------- parameter estimate ----------

function fusionParams(op, cs, out) {
  const n = cs.length, sum = cs.reduce((a, c) => a + c, 0);
  const proj = cs.reduce((a, c) => a + (c === out ? 0 : c * out + out), 0);
  if (op === "concat") return sum * out + 2 * out;
  if (op === "add") return proj;
  if (op === "gated") return proj + n * out * n * 9 + n;
  if (op === "attention") { const h = Math.max(8, Math.floor(sum / 4)); return 2 * sum * h + h + sum + sum * out + 2 * out; }
  if (op === "diff") return proj;
  if (op === "concat_diff") return proj + 3 * out * out + 2 * out;
  const dim = Math.min(256, out);
  return cs[0] * dim + (sum - cs[0]) * 2 * dim + dim * out + (cs[0] === out ? 0 : cs[0] * out);
}

function gcd(a, b) { return b ? gcd(b, a % b) : a; }

function temporalParams(r, info) {   // per-level aggregation modules of a time-series encoder
  return info.channels.reduce((a, c) => {
    if (r.temporal === "attention") { const h = gcd(c, 4), k = h * 16; return a + c * k + k + 32 * k + k + k; }
    if (r.temporal === "concat") return a + r.dates * c * c + 2 * c;
    return a;
  }, 0);
}

function metaWidth(m) {
  return m.fields.reduce((a, f) => a + ({ cyclic: 2, coords: 3, categorical: 8 }[f.type] ?? 1), 0);
}

function estimate(s, D) {
  const parts = { encoders: 0, fusion: 0, decoders: 0, heads: 0, metadata: 0 };
  for (const e of D.encoders) if (!e.shared) parts.encoders += e.info.params_m * 1e6;
  for (const e of D.encoders) if (e.inputs.length === 1 && e.inputs[0].kind === "timeseries") parts.encoders += temporalParams(e.inputs[0], e.info);
  if (D.ftype === "intermediate") {
    const p = D.primary;
    for (const st of D.fused) {
      const out = p.channels[p.strides.indexOf(st)];
      const cs = D.encoders.map((e) => e.info.channels[e.info.strides.indexOf(st)]).filter((c) => c);
      parts.fusion += fusionParams(s.fusion.op, cs, out);
    }
  }
  const nDec = D.ftype === "late" ? D.encoders.length : 1;
  if (D.shared && !D.shared.error) parts.decoders += D.shared.params_m * 1e6 * nDec;
  for (const t of Object.values(D.task)) if (!t.error && !t.sharesWith) parts.decoders += t.params_m * 1e6 * (t.input ? 1 : nDec);
  if (D.ftype === "late" && D.shared && !D.shared.error)
    parts.fusion += fusionParams(s.fusion.op, D.encoders.map(() => D.shared.channels), D.shared.channels);
  const deep = D.primary.channels[D.primary.channels.length - 1];
  for (const h of s.heads) {
    const c = h.channels * (h.uncertainty ? 2 : 1);
    if (h.scope === "image") parts.heads += deep * c + c;
    else {
      const src = (h.decoder || h.input) ? D.task[h.name] : D.shared;
      if (src && !src.error && !src.sharesWith) parts.heads += src.channels * c * 9 + c;
    }
  }
  if (s.meta && s.meta.fields.length) {
    const dim = 64, w = metaWidth(s.meta);
    parts.metadata += w * dim + dim + dim * dim + dim;
    if (s.meta.mode === "film") {
      const levels = D.primary.channels.slice();
      if (D.shared && !D.shared.error) levels.push(D.shared.channels);
      parts.metadata += levels.reduce((a, c) => a + dim * 2 * c + 2 * c, 0);
    }
  }
  parts.total = Object.values(parts).reduce((a, v) => a + v, 0);
  return parts;
}

function fmtParams(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e5 ? 0 : 1)}k`;
  return `${Math.round(n)}`;
}

// ---------- shapes ----------

// the example's primary size: a fixed-size encoder must not see more than its native size, and an
// input finer than the primary (relative pixel size < 1) is larger than the primary by 1/gsd
function clampForFixed(s, encoders) {
  let [H, W] = s.size;
  for (const e of encoders) {
    if (e.info.size !== "fixed") continue;
    const gsd = e.inputs.length === 1 ? e.inputs[0].gsd : 1;
    const lim = Math.max(1, Math.floor(FIXED_NATIVE * gsd));
    H = Math.min(H, lim); W = Math.min(W, lim);
  }
  return [H, W];
}

function sampleSize(s, D) {
  return clampForFixed(s, D.encoders);
}

// one plain segmentation head: create_model(num_classes=...) and the model returns a tensor
function isSingleOutput(s) {
  const h = s.heads[0];
  return s.heads.length === 1 && h.type === "segmentation" && h.scope === "pixel" && h.scale === 1 &&
    !h.decoder && !h.input && !h.uncertainty;
}

function outputShapes(s, D) {
  const [H, W] = sampleSize(s, D), B = s.batch;
  return s.heads.map((h) => {
    const c = h.channels * (h.uncertainty ? 2 : 1);
    if (h.scope === "image") return { name: h.name, shape: [B, c] };
    return { name: h.name, shape: [B, c, Math.ceil(H * h.scale), Math.ceil(W * h.scale)] };
  });
}

// ---------- Python code ----------

function py(v) {
  if (v === true) return "True";
  if (v === false) return "False";
  if (v === null || v === undefined) return "None";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(v);
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(py).join(", ")}]`;
  return `{${Object.entries(v).map(([k, x]) => `${JSON.stringify(k)}: ${py(x)}`).join(", ")}}`;
}

function pyBlock(obj, indent) {   // one entry per line; nested dicts expand when too long
  const pad = " ".repeat(indent + 4);
  const lines = Object.entries(obj).map(([k, v]) => {
    const head = `${pad}${JSON.stringify(k)}: `;
    const inline = py(v);
    if (head.length + inline.length < 96 || typeof v !== "object" || Array.isArray(v)) return `${head}${inline},`;
    return `${head}${pyBlock(v, indent + 4)},`;
  });
  return `{\n${lines.join("\n")}\n${" ".repeat(indent)}}`;
}

function matchingPreset(s, D) {
  if (D.multi || s.meta || !D.needsShared || Object.keys(D.task).length) return null;
  if (s.inputs[0].backbone !== "pvt_v2_b0.in1k" || s.decoder.out_stride !== 1 || s.decoder.input_cross) return null;
  for (const [name, p] of Object.entries(DATA.presets)) {
    const kw = p.kwargs;
    const width = kw.decoder_kwargs ? Object.entries(DATA.unet_widths).find(([, v]) => v && v.join() === kw.decoder_kwargs.channels.join())?.[0] : "full";
    if (kw.decoder === s.decoder.type && (kw.decoder !== "unet" || width === s.decoder.width)) return name;
  }
  return null;
}

function decoderKwargs(d, deepest) {
  const kw = {};
  if (UNET_FAMILY.includes(d.type) && d.out_stride !== 1) kw.out_stride = d.out_stride;
  if (d.type === "unet" && d.width !== "full") {
    const nblocks = Math.log2(deepest / d.out_stride);
    kw.channels = DATA.unet_widths[d.width].slice(0, nblocks);
  }
  if (d.type === "unet" && d.input_cross && d.out_stride === 1) kw.input_cross = true;
  return kw;
}

function metaConfig(m, hasImageHeads) {
  const fields = {};
  for (const f of m.fields) {
    const c = { type: f.type };
    if (f.type === "cyclic") c.period = f.period ?? 365.25;
    if (f.type === "continuous") { c.mean = f.mean ?? 0; c.std = f.std ?? 1; }
    if (f.type === "categorical") c.num = f.num ?? 4;
    fields[f.name] = c;
  }
  const cfg = { kind: "vector", fields };
  // 'head' with no image heads must also condition the pixel heads, or ManySeg rejects it
  if (m.mode === "head" && !hasImageHeads) cfg.conditioning = { mode: "head", pixel: true };
  else if (m.mode && m.mode !== "film") cfg.conditioning = m.mode;
  return cfg;
}

function metaColumns(m) {
  return m.fields.reduce((a, f) => a + (f.type === "coords" ? 2 : 1), 0);
}

function generateCode(s, D) {
  const args = [];
  const imports = ["import torch", "from manyseg import create_model"];
  const preset = matchingPreset(s, D);
  const useInputs = D.useInputs;
  const single = isSingleOutput(s);
  const pretrainedValue = (r) => r.weights === "none" ? false : r.weights === "imagenet" ? true : r.weights;
  const perInputWeights = D.multi && D.ftype !== "early";

  if (preset) {
    imports.push("from manyseg.presets import preset");
    args.push(`**preset(${py(preset)})`);
  } else if (!useInputs || D.ftype === "early" || !D.multi) {
    if (s.inputs[0].backbone !== "resnet34" || !useInputs) args.push(`backbone=${py(s.inputs[0].backbone)}`);
  }
  if (!perInputWeights && pretrainedValue(s.inputs[0]) !== true) args.push(`pretrained=${py(pretrainedValue(s.inputs[0]))}`);
  if (s.channel_adapter !== "repeat") args.push(`channel_adapter=${py(s.channel_adapter)}`);

  if (!useInputs) {
    if (s.inputs[0].channels !== 3) args.push(`in_channels=${s.inputs[0].channels}`);
  } else {
    const inputs = {};
    s.inputs.forEach((r, i) => {
      const c = {};
      if (r.kind === "timeseries") {
        c.kind = "timeseries";
        if (r.temporal === "concat") c.temporal = { type: "concat", dates: r.dates };
        else if (r.temporal !== "mean") c.temporal = r.temporal;
      }
      c.in_channels = r.channels;
      if (perInputWeights) {
        c.backbone = r.backbone;
        if (r.backbone.startsWith("shared:")) { if (r.share_norm) c.share_norm = true; }
        else if (pretrainedValue(r) !== true) c.pretrained = pretrainedValue(r);
      }
      if (i > 0 && r.alignment === "same_extent") c.alignment = "same_extent";
      inputs[r.name] = c;
    });
    if (s.meta) inputs[s.meta.name] = metaConfig(s.meta, D.imageHeads.length > 0);
    args.push(`inputs=${pyBlock(inputs, 4)}`);
    if (D.multi) {
      const f = { type: D.ftype };
      if (D.ftype !== "early") {
        if (s.fusion.op !== "concat" || D.ftype === "late") f.op = s.fusion.op;
        if (D.ftype === "intermediate" && s.fusion.op === "cross_attention") f.strides = D.fused;
      }
      args.push(`fusion=${py(f)}`);
    }
  }

  if (D.needsShared && !preset) {
    if (s.decoder.type !== "unet") args.push(`decoder=${py(s.decoder.type)}`);
    const kw = decoderKwargs(s.decoder, Math.max(...D.primary.strides));
    if (Object.keys(kw).length) args.push(`decoder_kwargs=${py(kw)}`);
  }

  if (single) {
    args.push(`num_classes=${s.heads[0].channels}`);
  } else {
    const heads = {};
    for (const h of s.heads) {
      const c = { type: h.type, out_channels: h.channels };
      if (h.scope === "image") c.scope = "image";
      if (h.scope === "pixel" && h.scale !== 1) c.scale = h.scale;
      if (h.uncertainty && h.type === "regression") c.uncertainty = true;
      if (h.scope === "pixel" && h.decoder && !h.share) c.decoder = h.decoder;
      if (h.input) c.input = h.input;
      if (h.input && h.share) c.share = h.share;
      heads[h.name] = c;
    }
    args.push(`heads=${pyBlock(heads, 4)}`);
  }

  const lines = [...imports, "", "model = create_model(", ...args.map((a) => `    ${a},`), ")", ""];

  // example forward pass
  const [H, W] = sampleSize(s, D), B = s.batch;
  if (!useInputs) {
    lines.push(`x = torch.randn(${B}, ${s.inputs[0].channels}, ${H}, ${W})`);
  } else {
    lines.push("x = {");
    const mixed = s.inputs.some((r) => r.res !== 1);
    for (const r of s.inputs) {
      const h = inputPixels(s, r, H), w = inputPixels(s, r, W);
      const note = mixed ? `  # ${r.res === 1 ? "primary grid" : resLabel(r.res)}` : "";
      if (r.kind === "timeseries") {
        const T = r.dates;
        lines.push(`    ${py(r.name)}: torch.randn(${B}, ${T}, ${r.channels}, ${h}, ${w}),  # (B, T, C, H, W)${mixed ? `, ${r.res === 1 ? "primary grid" : resLabel(r.res)}` : ""}`);
        lines.push(`    ${py(r.name + "_mask")}: torch.ones(${B}, ${T}, dtype=torch.bool),  # optional: False = missing date`);
        if (r.temporal === "attention")
          lines.push(`    ${py(r.name + "_dates")}: torch.linspace(0, 300, ${T}).expand(${B}, ${T}),  # optional: day of year`);
      } else {
        lines.push(`    ${py(r.name)}: torch.randn(${B}, ${r.channels}, ${h}, ${w}),${note}`);
      }
    }
    if (s.meta) {
      const cols = s.meta.fields.map((f) => f.type === "coords" ? `${f.name} (lat, lon)` : f.name).join(", ");
      lines.push(`    ${py(s.meta.name)}: torch.rand(${B}, ${metaColumns(s.meta)}),  # columns: ${cols}`);
    }
    lines.push("}");
  }
  lines.push("out = model(x)");
  const shapes = outputShapes(s, D);
  if (single) lines.push(`# out: (${shapes[0].shape.join(", ")})`);
  else for (const o of shapes) lines.push(`# out[${py(o.name)}]: (${o.shape.join(", ")})`);
  lines.push("print(model.describe(x))  # what was built, with shapes and FLOPs");
  // after training: manyseg.save stores the weights plus what inference needs; activations are
  // required for segmentation and classification heads (softmax, or sigmoid for one channel)
  const acts = {};
  for (const h of s.heads) if (h.type !== "regression") acts[single ? "logits" : h.name] = h.channels === 1 ? "sigmoid" : "softmax";
  const rs = s.inputs.filter((r) => !r.backbone.startsWith("shared:") && !(D.ftype === "early" && r !== s.inputs[0]));
  const rsWeights = rs.length && rs.every((r) => !["imagenet", "none"].includes(r.weights));
  lines.push("");
  lines.push("# After training: one file with the weights and what inference needs.");
  if (!rsWeights) lines.push("# Add normalize=[...] (the normalisation you trained with); see help(manyseg.save).");
  lines.push("# import manyseg");
  const saveArgs = [`"model.safetensors"`];
  if (Object.keys(acts).length) saveArgs.push(`activations=${py(acts)}`);
  if (rsWeights) saveArgs.push(`normalize="weights"`);
  lines.push(`# manyseg.save(model, ${saveArgs.join(", ")})`);
  lines.push("# model = manyseg.load(\"model.safetensors\")  # model.tiling_spec(): sizes for tiled inference");
  if (s.meta) lines.push(`# real metadata: model.pack_metadata(${py(s.meta.name)}, {...})`);
  return lines.join("\n");
}

if (typeof module !== "undefined") {
  module.exports = { EXAMPLES, HEAD_GROUPS, makeState, syncRes, resLabel, derive, estimate, generateCode, outputShapes, fmtParams, weightTags, isSingleOutput };
}
