// ManySeg model builder: form, live diagram and code panel.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const INPUT_COLORS = ["var(--in1)", "var(--in2)", "var(--in3)", "var(--in4)"];
const STORE_KEY = "manyseg-builder-v1";

let state, activeExample = "fusion";
try {
  const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
  if (saved && saved.state) { state = makeState(saved.state); activeExample = saved.example ?? null; }
} catch (e) { /* storage unavailable: start from the example */ }
if (!state) state = makeState(EXAMPLES.fusion.state);
const hashEx = (location.hash || "").slice(1);
if (EXAMPLES[hashEx]) { state = makeState(EXAMPLES[hashEx].state); activeExample = hashEx; }

function save() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify({ state, example: activeExample })); } catch (e) { /* ignore */ }
}

// ---------- form ----------

function opt(value, label, sel) {
  return `<option value="${esc(value)}"${String(value) === String(sel) ? " selected" : ""}>${esc(label)}</option>`;
}

function backboneOptions(sel, idx) {
  let html = "";
  if (idx > 0 && state.fusion.type !== "early") {
    html += opt(`shared:${state.inputs[0].name}`, `share ${state.inputs[0].name}'s encoder weights`, sel);
  }
  html += `<optgroup label="Default and preset encoders">`;
  for (const b of DATA.backbones.slice(0, 2)) html += opt(b.name, `${b.name}  \u00b7  ${fmtParams(b.params_m * 1e6)}`, sel);
  html += `</optgroup><optgroup label="Tested timm encoders">`;
  for (const b of DATA.backbones.slice(2)) {
    const flag = b.size === "fixed" ? "  \u00b7 fixed size" : b.vit ? "  \u00b7 ViT" : "";
    html += opt(b.name, `${b.name}  \u00b7  ${fmtParams(b.params_m * 1e6)}${flag}`, sel);
  }
  return html + "</optgroup>";
}

function info(key, what) {
  return `<button type="button" class="info" data-tipk="${key}" aria-label="About ${esc(what)}">i</button>`;
}

function lab(text, key) {   // a field label with an info button
  return `<span class="lt">${text}${key ? info(key, text) : ""}</span>`;
}

function segmented(path, value, options, disabledFn = () => false, tipPrefix = "") {
  return `<div class="seg" role="group">${options.map(([v, l]) =>
    `<button type="button" data-set="${path}" data-val="${esc(JSON.stringify(v))}" aria-pressed="${v === value}"${tipPrefix ? ` data-tipk="${tipPrefix}.${v}"` : ""}${disabledFn(v) ? " disabled" : ""}>${esc(l)}</button>`).join("")}</div>`;
}

function renderForm() {
  const s = state, D = derive(s);
  const multi = s.inputs.length > 1;
  let h = "";

  // inputs
  h += `<section class="sec"><h2>Inputs${info("sec.inputs", "inputs")}${multi ? `<span class="note">first input is the primary grid</span>` : ""}</h2>`;
  // pixel sizes only matter once an input covers the same extent at another resolution
  const showRes = s.inputs.some((r, i) => i > 0 && r.alignment === "same_extent");
  s.inputs.forEach((r, i) => {
    const early = multi && s.fusion.type === "early";
    h += `<div class="card" style="--c:${INPUT_COLORS[i % 4]}">
      <div class="cardhead"><b>${i === 0 ? "Primary raster" : "Raster"}</b>${i > 0 ? `<button class="x" type="button" data-act="rm-input" data-i="${i}">Remove</button>` : ""}</div>
      <div class="row" style="${multi ? `--cols:3; grid-template-columns: 1.3fr .8fr 1.2fr` : `--cols:2; grid-template-columns: 1.3fr .7fr`}">
        <label class="f">Name<input type="text" class="mono" id="in-${i}-name" data-path="inputs.${i}.name" value="${esc(r.name)}"></label>
        <label class="f">${lab("Channels", "f.channels")}<input type="number" min="1" max="512" id="in-${i}-ch" data-path="inputs.${i}.channels" data-type="int" value="${r.channels}"></label>
        ${!multi ? "" : i === 0 ? `<label class="f">${lab("Grid", "f.grid")}<select id="in-0-al" disabled><option>primary</option></select></label>` :
        `<label class="f">${lab("Grid", "f.grid")}<select id="in-${i}-al" data-path="inputs.${i}.alignment" data-type="grid"${early ? " disabled" : ""}>
          ${opt("same_grid", "same grid", r.alignment)}${opt("same_extent", "same extent", r.alignment)}
        </select></label>`}

      </div>
      ${(() => {
        const enc = early && i > 0 ? "" : `<label class="f">${lab(early ? "Encoder (shared by all stacked inputs)" : "Encoder", "f.encoder")}<select class="mono" id="in-${i}-bb" data-path="inputs.${i}.backbone">${backboneOptions(r.backbone, i)}</select></label>`;
        if (!(multi && showRes && i > 0 && r.alignment === "same_extent")) return enc;
        const g = `<label class="f">${lab("Rel. pixel size", "f.gsd")}<input type="number" class="nospin" inputmode="decimal" min="0" step="any" id="in-${i}-gsd" data-path="inputs.${i}.gsd" data-type="float" value="${r.gsd}"></label>`;
        return `<div class="row" style="--cols:2; grid-template-columns: minmax(0, 1fr) 118px">${enc || "<span></span>"}${g}</div>`;
      })()}
      ${(() => {
        // data kind, then pretrained weights (own encoder) or BatchNorm sharing (shared encoder)
        const shared = r.backbone.startsWith("shared:");
        const kind = `<label class="f">${lab("Data", "f.kind")}<select id="in-${i}-kind" data-path="inputs.${i}.kind">${opt("raster", "image (B, C, H, W)", r.kind)}${opt("timeseries", "time series (B, T, C, H, W)", r.kind)}</select></label>`;
        let second = "<span></span>";
        if (shared) second = `<label class="check" data-tipk="f.sharenorm"><input type="checkbox" id="in-${i}-sn" data-path="inputs.${i}.share_norm" data-type="bool"${r.share_norm ? " checked" : ""}> share BatchNorm too</label>`;
        else if (!(early && i > 0)) {
          const tags = weightTags(r.backbone);
          second = `<label class="f">${lab("Weights", "f.weights")}<select id="in-${i}-w" data-path="inputs.${i}.weights">${opt("imagenet", "ImageNet (timm)", r.weights)}${opt("none", "random init", r.weights)}${tags.map((w) => opt(w.tag, `${w.tag} \u00b7 ${w.bands} ch`, r.weights)).join("")}</select></label>`;
        }
        let ts = "";
        if (r.kind === "timeseries") {
          ts = `<div class="row">
            <label class="f">${lab("Aggregation", "f.temporal")}<select id="in-${i}-agg" data-path="inputs.${i}.temporal">${(DATA.temporal || ["mean"]).map((a) => opt(a, a, r.temporal)).join("")}</select></label>
            <label class="f">${lab(r.temporal === "concat" ? "Dates (fixed)" : "Dates in example", "f.dates")}<input type="number" min="1" max="366" id="in-${i}-T" data-path="inputs.${i}.dates" data-type="int" value="${r.dates}"${r.temporal === "diff" ? " disabled" : ""}></label>
          </div>`;
        }
        return `<div class="row">${kind}${second}</div>${ts}`;
      })()}
    </div>`;
  });
  if (s.meta) {
    const m = s.meta;
    h += `<div class="card" style="--c:var(--meta)">
      <div class="cardhead"><b>Metadata vector</b>${info("sec.meta", "metadata")}<button class="x" type="button" data-act="rm-meta">Remove</button></div>
      <div class="row">
        <label class="f">Name<input type="text" class="mono" id="meta-name" data-path="meta.name" value="${esc(m.name)}"></label>
        <label class="f">${lab("Injection", "f.inject")}<select id="meta-mode" data-path="meta.mode">${opt("film", "FiLM on decoder levels", m.mode)}${opt("head", "bias on image heads", m.mode)}${opt("channels", "extra input channels", m.mode)}</select></label>
      </div>`;
    m.fields.forEach((f, j) => {
      let extra = `<span></span>`;
      if (f.type === "cyclic") extra = `<label class="f">Period<input type="number" id="mf-${j}-p" data-path="meta.fields.${j}.period" data-type="float" value="${f.period ?? 365.25}"></label>`;
      if (f.type === "continuous") extra = `<div class="row" style="--cols:2;gap:4px"><label class="f">Mean<input type="number" id="mf-${j}-m" data-path="meta.fields.${j}.mean" data-type="float" value="${f.mean ?? 0}"></label><label class="f">Std<input type="number" id="mf-${j}-s" data-path="meta.fields.${j}.std" data-type="float" value="${f.std ?? 1}"></label></div>`;
      if (f.type === "categorical") extra = `<label class="f">Categories<input type="number" min="1" id="mf-${j}-n" data-path="meta.fields.${j}.num" data-type="int" value="${f.num ?? 4}"></label>`;
      if (f.type === "coords") extra = `<p class="hint">lat, lon in degrees</p>`;
      h += `<div class="field-row">
        <label class="f">Field<input type="text" class="mono" id="mf-${j}-name" data-path="meta.fields.${j}.name" value="${esc(f.name)}"></label>
        <label class="f">${lab("Type", "f.ftype")}<select id="mf-${j}-t" data-path="meta.fields.${j}.type">${DATA.field_types.map((t) => opt(t, t, f.type)).join("")}</select></label>
        ${extra}
        <button class="x" type="button" data-act="rm-field" data-i="${j}" aria-label="Remove field">\u2715</button>
      </div>`;
    });
    h += `<div class="btns" style="margin-top:2px"><button class="btn" type="button" data-act="add-field">+ Field</button></div></div>`;
  }
  h += `<div class="btns">${s.inputs.length < 4 ? `<button class="btn" type="button" data-act="add-input">+ Raster input</button>` : ""}${s.meta ? "" : `<button class="btn" type="button" data-act="add-meta">+ Metadata vector (date, sensor, age\u2026)</button>`}</div></section>`;

  // fusion
  if (multi) {
    const ft = s.fusion.type;
    h += `<section class="sec"><h2>Fusion${info("sec.fusion", "fusion")}</h2>
      <div class="row" style="--cols:1">${segmented("fusion.type", ft, [["early", "Early: stack channels"], ["intermediate", "Intermediate"], ["late", "Late"]], () => false, "fusion")}</div>
      <p class="hint" style="margin-top:8px">${{ early: "All inputs are stacked on the channel axis into one encoder. Inputs must share a grid.",
        intermediate: "Each input has its own encoder; features at matching strides are fused, then one decoder runs.",
        late: "Each input gets its own encoder and decoder; the decoder outputs are fused before the heads." }[ft]}</p>
      ${ft === "early" ? "" : `<div class="row" style="margin-top:8px"><label class="f">${lab("Fusion op", "f.fop")}<select id="fu-op" data-path="fusion.op">${DATA.fusion_ops.map((o) => opt(o, o.replace("_", " "), s.fusion.op)).join("")}</select></label><span></span></div>`}
    </section>`;
  }

  // decoder
  const d = s.decoder;
  h += `<section class="sec"><h2>Decoder${info("sec.decoder", "the decoder")}${D.needsShared ? "" : `<span class="note">not built</span>`}</h2>`;
  if (!D.needsShared) {
    h += `<p class="hint">${D.pixelHeads.length ? "Every pixel head has its own decoder, so no shared decoder is built." : "Only image-level heads: they pool the deepest encoder feature and no decoder is built."}</p>`;
  } else {
    const unetFam = ["unet", "unetplusplus", "fastai_unet"].includes(d.type);
    const presetNow = (() => { const D2 = derive(s); return matchingPresetName(s, D2); })();
    h += `<div class="row">
        <label class="f">${lab("Type", "f.dtype")}<select class="mono" id="dec-type" data-path="decoder.type">${DATA.decoders.map((t) => opt(t, t, d.type)).join("")}</select></label>
        <label class="f">${lab("Speed preset", "f.preset")}<select id="dec-preset" data-act-change="preset">${opt("", presetNow ? presetNow : "custom", "")}${Object.entries(DATA.presets).map(([k, p]) => opt(k, `${k} \u00b7 ${p.measured.latency_ms} ms`, "")).join("")}</select></label>
      </div>
      ${unetFam ? `<div class="row" style="--cols:1"><label class="f">${lab("Output stride (1 = full resolution)", "")}
        ${segmented("decoder.out_stride", d.out_stride, [[1, "1"], [2, "2"], [4, "4"]], () => false, "dec.os")}</label></div>` : ""}`;
    if (d.type === "unet") {
      h += `<div class="row"><label class="f">${lab("Width", "")}
          ${segmented("decoder.width", d.width, [["full", "full"], ["half", "\u00bd"], ["quarter", "\u00bc"]], () => false, "dec.w")}</label>
        <label class="check" data-tipk="dec.ic"><input type="checkbox" id="dec-ic" data-path="decoder.input_cross" data-type="bool"${d.input_cross ? " checked" : ""}${d.out_stride !== 1 || multi || s.inputs.some((r) => r.kind === "timeseries") ? " disabled" : ""}> input_cross</label></div>`;
    }
    if (!unetFam) h += `<p class="hint">${d.type} outputs at the encoder's finest level${D.shared && !D.shared.error ? ` (/${D.shared.stride} for ${D.primary.name})` : ""}; pixel heads upsample.</p>`;
  }
  h += `</section>`;

  // heads
  h += `<section class="sec"><h2>Heads${info("sec.heads", "heads")}<span class="note">outputs, by name</span></h2>`;
  s.heads.forEach((hd, i) => {
    const pix = hd.scope === "pixel";
    h += `<div class="card" style="--c:var(--line-strong)">
      <div class="row" style="--cols:4; grid-template-columns: 1.3fr 1.3fr .8fr 1fr">
        <label class="f">Name<input type="text" class="mono" id="hd-${i}-name" data-path="heads.${i}.name" value="${esc(hd.name)}"></label>
        <label class="f">${lab("Type", "f.htype")}<select id="hd-${i}-type" data-path="heads.${i}.type">${opt("segmentation", "segmentation", hd.type)}${opt("classification", "classification", hd.type)}${opt("regression", "regression", hd.type)}</select></label>
        <label class="f">${lab(hd.type === "regression" ? "Values" : "Classes", "f.hch")}<input type="number" min="1" id="hd-${i}-ch" data-path="heads.${i}.channels" data-type="int" value="${hd.channels}"></label>
        <label class="f">${lab("Per", "f.scope")}<select id="hd-${i}-scope" data-path="heads.${i}.scope">${opt("pixel", "pixel", hd.scope)}${opt("image", "image", hd.scope)}</select></label>
      </div>
      <div class="row" style="--cols:4; grid-template-columns: 1fr 1.3fr 1fr auto">
        ${pix ? `<label class="f">${lab("Scale", "f.scale")}<select id="hd-${i}-scale" data-path="heads.${i}.scale" data-type="float">${opt(1, "1", hd.scale)}${opt(0.5, "\u00bd", hd.scale)}${opt(0.25, "\u00bc", hd.scale)}${opt(0.125, "\u215b", hd.scale)}</select></label>
        <label class="f">${lab("Decoder", "f.hdec")}<select id="hd-${i}-dec" data-path="heads.${i}.decoder"${hd.share ? " disabled" : ""}>${opt("", hd.input ? `own ${s.decoder.type} (model's)` : "shared", hd.decoder)}${DATA.decoders.map((t) => opt(t, `own ${t}`, hd.decoder)).join("")}</select></label>` : `<span class="hint" style="grid-column: span 2; align-self:center">pools the deepest encoder feature</span>`}
        ${hd.type === "regression" ? `<label class="check" data-tipk="h.unc"><input type="checkbox" id="hd-${i}-unc" data-path="heads.${i}.uncertainty" data-type="bool"${hd.uncertainty ? " checked" : ""}> uncertainty</label>` : "<span></span>"}
        <button class="x" type="button" data-act="rm-head" data-i="${i}">Remove</button>
      </div>
      ${(() => {
        // per-input heads: intermediate fusion of several rasters (semantic change, per-date maps)
        if (!(pix && multi && s.fusion.type === "intermediate")) return "";
        const reads = `<label class="f">${lab("Reads", "f.reads")}<select id="hd-${i}-in" data-path="heads.${i}.input">${opt("", "fused features", hd.input)}${s.inputs.map((r) => opt(r.name, `input ${r.name} only`, hd.input)).join("")}</select></label>`;
        if (!hd.input) return `<div class="row">${reads}<span></span></div>`;
        const others = s.heads.filter((o) => o !== hd && o.input && !o.share && o.scope === "pixel");
        const share = `<label class="f">${lab("Shares weights with", "f.share")}<select id="hd-${i}-sh" data-path="heads.${i}.share">${opt("", "nothing", hd.share)}${others.map((o) => opt(o.name, o.name, hd.share)).join("")}</select></label>`;
        return `<div class="row">${reads}${share}</div>`;
      })()}
      ${pix ? headLayers(hd, i) : ""}
    </div>`;
  });
  h += `<div class="btns"><button class="btn" type="button" data-act="add-head">+ Head</button>
    <span class="hint" style="margin-left:6px">Add a set:</span>
    <button class="btn" type="button" data-act="group" data-g="instances">instances</button>
    <button class="btn" type="button" data-act="group" data-g="points">points</button>
    <button class="btn" type="button" data-act="group" data-g="density">density</button>
    <button class="btn" type="button" data-act="group" data-g="density_softplus" data-tipk="g.densp">density, softplus</button></div></section>`;

  // options
  h += `<section class="sec"><h2>Options${info("sec.options", "options")}</h2>
    <div class="row" style="--cols:2">
      <span></span>
      <label class="f">${lab("Channel adapter", "f.adapter")}<select id="opt-ad" data-path="channel_adapter">${["repeat", "mean", "rgb_plus_random", "rgb_plus_zero", "projection"].map((a) => opt(a, a, s.channel_adapter)).join("")}</select></label>
    </div>
    <div class="row" style="--cols:3; margin-top:8px">
      <label class="f">${lab("Example height", "f.size")}<input type="number" min="32" max="4096" id="opt-h" data-path="size.0" data-type="int" value="${s.size[0]}"></label>
      <label class="f">Example width<input type="number" min="32" max="4096" id="opt-w" data-path="size.1" data-type="int" value="${s.size[1]}"></label>
      <label class="f">Batch<input type="number" min="1" max="256" id="opt-b" data-path="batch" data-type="int" value="${s.batch}"></label>
    </div></section>`;

  if (typeof hideTip === "function") hideTip();
  const focused = document.activeElement && document.activeElement.id;
  $("form").innerHTML = h;
  if (focused && $(focused)) $(focused).focus();
}

// PixelHead options (hidden layers, output activation, bias), folded away until used.
// headOpen: index -> open, once the user toggles it; otherwise open when an option is set.
const headOpen = new Map();
function headLayers(hd, i) {
  const { opts, hidden } = headOptions(hd);
  const set = Object.keys(opts).length || hd.hidden.trim();
  const open = headOpen.has(i) ? headOpen.get(i) : !!set;
  const bits = [];
  if (hidden.length) bits.push(`hidden ${hidden.join(", ")}`);
  if ("dilation" in opts) bits.push(`dilation ${listText(opts.dilation)}`);
  if ("norm" in opts) bits.push(opts.norm ? `${opts.norm} norm` : "no norm");
  if (opts.activation) bits.push(opts.activation);
  if (opts.bias_init) bits.push(`bias ${opts.bias_init}`);
  const off = hidden.length ? "" : " disabled";
  return `<details class="more" data-hd="${i}"${open ? " open" : ""}>
      <summary>${lab("Head layers", "f.hlayers")}<span class="hint">${esc(bits.length ? bits.join(" \u00b7 ") : "single 3\u00d73 conv")}</span></summary>
      <div class="row" style="--cols:3">
        <label class="f">${lab("Hidden widths", "f.hidden")}<input type="text" class="mono" id="hd-${i}-hid" data-path="heads.${i}.hidden" placeholder="none, e.g. 64, 32" value="${esc(hd.hidden)}"></label>
        <label class="f">${lab("Dilation", "f.dil")}<input type="text" class="mono" id="hd-${i}-dil" data-path="heads.${i}.dilation" placeholder="1" value="${esc(hd.dilation)}"${off}></label>
        <label class="f">${lab("Norm", "f.hnorm")}<select id="hd-${i}-norm" data-path="heads.${i}.norm"${off}>${opt("bn", "BatchNorm", hd.norm)}${opt("gn", "GroupNorm", hd.norm)}${opt("none", "none", hd.norm)}</select></label>
      </div>
      <div class="row" style="--cols:3; margin-top:8px">
        ${hd.type === "regression" ? `<label class="f">${lab("Output", "f.hact")}<select id="hd-${i}-act" data-path="heads.${i}.activation">${opt("", "raw values", hd.activation)}${opt("softplus", "softplus (> 0)", hd.activation)}</select></label>` : "<span></span>"}
        <label class="f">${lab("Bias init", "f.hbias")}<input type="number" step="any" class="nospin" id="hd-${i}-bias" data-path="heads.${i}.bias_init" data-type="float" value="${hd.bias_init}"></label>
        <span></span>
      </div>
    </details>`;
}

function matchingPresetName(s, D) {
  try { return (typeof matchingPreset === "function") ? matchingPreset(s, D) : null; } catch (e) { return null; }
}

function weightsSummary(s) {
  // early fusion has one encoder, initialised from the primary input's setting
  const own = s.inputs.length > 1 && s.fusion.type === "early" ? [s.inputs[0]]
    : s.inputs.filter((r) => !r.backbone.startsWith("shared:"));
  const ws = new Set(own.map((r) => r.weights));
  if (ws.size > 1) return "mixed pretrained weights";
  const w = [...ws][0];
  return w === "imagenet" ? "ImageNet weights, channel-adapted" : w === "none" ? "random init" : `${w} weights`;
}

function renderExamples() {
  $("examples").innerHTML = `<span class="lab">Start from</span>` + Object.entries(EXAMPLES).map(([k, e]) =>
    `<button class="chip" type="button" data-ex="${k}" aria-pressed="${k === activeExample}">${esc(e.label)}</button>`).join("") +
    `<div class="blurb">${activeExample ? esc(EXAMPLES[activeExample].blurb) : "Custom configuration."}</div>`;
}

// ---------- state updates ----------

function setPath(path, raw, type) {
  const keys = path.split(".");
  let obj = state;
  for (const k of keys.slice(0, -1)) obj = obj[k];
  const last = keys[keys.length - 1];
  let v = raw;
  if (type === "int") v = parseInt(raw, 10);
  if (type === "float") v = parseFloat(raw);
  if (type === "bool") v = !!raw;
  if (type === "grid") {
    obj.alignment = raw;
    if (raw === "same_extent" && obj.gsd === 1) obj.gsd = 2;
    syncRes(state);
    return;
  }
  if ((type === "int" || type === "float") && Number.isNaN(v)) return;
  const old = obj[last];
  obj[last] = v;
  // renames: keep every reference to the old name (shared encoders, per-input heads, shares)
  if (/^inputs\.\d+\.name$/.test(path) && old !== v) {
    state.inputs.forEach((r) => { if (r.backbone === `shared:${old}`) r.backbone = `shared:${v}`; });
    state.heads.forEach((h) => { if (h.input === old) h.input = v; });
  }
  if (/^heads\.\d+\.name$/.test(path) && old !== v) state.heads.forEach((h) => { if (h.share === old) h.share = v; });
  syncRes(state);
  // keep dependent options valid
  if (path === "decoder.type" && !["unet", "unetplusplus", "fastai_unet"].includes(v)) state.decoder.out_stride = 1;
  if (path === "decoder.out_stride" && v !== 1) state.decoder.input_cross = false;
  if (path === "fusion.type" && v === "early") state.inputs.forEach((r, i) => {
    r.alignment = "same_grid"; if (i > 0) r.backbone = state.inputs[0].backbone; });
  if (/^inputs\.0\.backbone$/.test(path) && state.fusion.type === "early") state.inputs.forEach((r) => { r.backbone = v; });
  if (/heads\.\d+\.type$/.test(path) && v !== "regression") { obj.uncertainty = false; obj.activation = ""; }
  // image heads have no head layers (create_head refuses them)
  if (/heads\.\d+\.scope$/.test(path) && v === "image") { obj.scale = 1; obj.decoder = ""; obj.input = ""; obj.share = ""; obj.hidden = ""; obj.activation = ""; obj.bias_init = 0; }
  if (/heads\.\d+\.input$/.test(path) && !v) obj.share = "";
  if (/heads\.\d+\.share$/.test(path) && v) obj.decoder = "";
  if (/inputs\.\d+\.temporal$/.test(path) && v === "diff") obj.dates = 2;
  if (/inputs\.\d+\.backbone$/.test(path) && !["imagenet", "none"].includes(obj.weights) && !weightTags(v).some((w) => w.tag === obj.weights)) obj.weights = "imagenet";
  if (/meta\.fields\.\d+\.type$/.test(path)) {
    if (v === "cyclic") obj.period = obj.period ?? 365.25;
    if (v === "continuous") { obj.mean = obj.mean ?? 0; obj.std = obj.std ?? 1; }
    if (v === "categorical") obj.num = obj.num ?? 4;
  }
}

function uniqueName(base, taken) {
  if (!taken.includes(base)) return base;
  for (let i = 2; ; i++) if (!taken.includes(`${base}${i}`)) return `${base}${i}`;
}

function act(a, el) {
  const s = state;
  if (a === "add-input") {
    const names = s.inputs.map((r) => r.name);
    const suggestions = ["sar", "dem", "lidar", "thermal"];
    const name = uniqueName(suggestions.find((n) => !names.includes(n)) || "input", names);
    s.inputs.push(normInput({ name, channels: 2, backbone: s.fusion.type === "early" ? s.inputs[0].backbone : "resnet18" }, s.inputs.length));
  }
  if (a === "rm-input") {
    const gone = s.inputs.splice(+el.dataset.i, 1)[0];
    s.inputs.forEach((r) => { if (r.backbone === `shared:${gone.name}`) r.backbone = "resnet18"; });
    s.heads.forEach((h) => { if (h.input === gone.name) { h.input = ""; h.share = ""; } });
  }
  if (a === "add-meta") s.meta = { name: "meta", mode: "film", fields: [{ name: "doy", type: "cyclic", period: 365.25 }] };
  if (a === "rm-meta") s.meta = null;
  if (a === "add-field") {
    const taken = s.meta.fields.map((f) => f.name);
    const next = [["sun_elev", "continuous"], ["latlon", "coords"], ["sensor", "categorical"], ["gsd", "continuous"]].find(([n]) => !taken.includes(n)) || [uniqueName("field", taken), "continuous"];
    s.meta.fields.push({ name: next[0], type: next[1], period: 365.25, mean: next[0] === "sun_elev" ? 45 : 0, std: next[0] === "sun_elev" ? 15 : 1, num: 3 });
  }
  if (a === "rm-field") { s.meta.fields.splice(+el.dataset.i, 1); if (!s.meta.fields.length) s.meta = null; }
  if (a === "add-head") {
    const taken = s.heads.map((h) => h.name);
    s.heads.push(normHead({ name: uniqueName("output", taken), type: "segmentation", channels: 2 }));
  }
  if (a === "rm-head") {
    headOpen.clear();   // indices shift
    const gone = s.heads.splice(+el.dataset.i, 1)[0];
    s.heads.forEach((h) => { if (h.share === gone.name) h.share = ""; });
  }
  if (a === "group") {
    const taken = s.heads.map((h) => h.name);
    for (const g of HEAD_GROUPS[el.dataset.g]) s.heads.push(normHead({ ...g, name: uniqueName(g.name, taken) }));
  }
}

function applyPreset(name) {
  const kw = DATA.presets[name].kwargs;
  state.inputs[0].backbone = kw.backbone;
  if (state.fusion.type === "early") state.inputs.forEach((r) => { r.backbone = kw.backbone; });
  state.decoder.type = kw.decoder;
  state.decoder.out_stride = 1;
  state.decoder.input_cross = false;
  state.decoder.width = kw.decoder_kwargs ? Object.entries(DATA.unet_widths).find(([, v]) => v && v.join() === kw.decoder_kwargs.channels.join())[0] : "full";
}

function changed(rebuildForm) {
  activeExample = null;
  syncRes(state);
  if (rebuildForm) renderForm();
  renderExamples();
  renderPreview();
  save();
}

document.addEventListener("input", (e) => {
  const el = e.target;
  if (!el.dataset || !el.dataset.path || el.tagName === "SELECT" || el.type === "checkbox") return;
  setPath(el.dataset.path, el.value, el.dataset.type);
  changed(false);
});
document.addEventListener("change", (e) => {
  const el = e.target;
  if (el.dataset.actChange === "preset") { if (el.value) applyPreset(el.value); changed(true); return; }
  if (!el.dataset || !el.dataset.path) return;
  if (el.tagName === "SELECT" || el.type === "checkbox") {
    setPath(el.dataset.path, el.type === "checkbox" ? el.checked : el.value, el.dataset.type);
    changed(true);
  } else changed(true);   // number/text commit: re-render dependent controls
});
document.addEventListener("toggle", (e) => {   // does not bubble, so capture
  const d = e.target;
  if (d.dataset && d.dataset.hd !== undefined) headOpen.set(+d.dataset.hd, d.open);
}, true);
document.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.ex) {
    state = makeState(EXAMPLES[b.dataset.ex].state);
    activeExample = b.dataset.ex;
    headOpen.clear();
    renderForm(); renderExamples(); renderPreview(); save();
  } else if (b.dataset.set) {
    setPath(b.dataset.set, JSON.parse(b.dataset.val));
    changed(true);
  } else if (b.dataset.act) {
    act(b.dataset.act, b);
    changed(true);
  } else if (b.id === "copy") {
    const text = $("code").textContent;
    const done = () => { b.textContent = "Copied"; setTimeout(() => { b.textContent = "Copy code"; }, 1400); };
    const fallback = () => { const r = document.createRange(); r.selectNodeContents($("code")); const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); b.textContent = "Selected: press Ctrl+C"; setTimeout(() => { b.textContent = "Copy code"; }, 2200); };
    try { navigator.clipboard.writeText(text).then(done, fallback); } catch (err) { fallback(); }
  }
});

// ---------- preview ----------

function highlight(code) {
  const re = /(#[^\n]*)|("(?:[^"\\]|\\.)*")|\b(import|from|True|False|None|print)\b|\b(\d+(?:\.\d+)?)\b|\b(create_model|preset)\b/g;
  let out = "", last = 0, m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    const cls = m[1] ? "c" : m[2] ? "s" : m[3] ? "k" : m[4] ? "n" : "fn";
    out += `<span class="${cls}">${esc(m[0])}</span>`;
    last = re.lastIndex;
  }
  return out + esc(code.slice(last));
}

function renderPreview() {
  const s = state, D = derive(s);
  const est = estimate(s, D);
  const errors = D.issues.filter((i) => i.level === "error");
  const single = isSingleOutput(s);

  const stat = (k, v, big) => `<div class="stat${big ? " big" : ""}"><span class="k">${k}</span><span class="v">${v}</span></div>`;
  $("stats").innerHTML = stat("Parameters", `\u2248 ${fmtParams(est.total)}`, true) +
    stat("Encoders", fmtParams(est.encoders)) +
    (est.fusion ? stat("Fusion", fmtParams(est.fusion)) : "") +
    stat("Decoders", est.decoders ? fmtParams(est.decoders) : "none") +
    stat("Heads", fmtParams(est.heads)) +
    (est.metadata ? stat("Metadata", fmtParams(est.metadata)) : "") +
    `<span class="src">${weightsSummary(s)}</span>`;

  $("diagram").innerHTML = drawDiagram(s, D);
  $("code").innerHTML = highlight(generateCode(s, D));
  $("returns").textContent = errors.length ? "Fix the errors below first" : single ? "returns a tensor" : `returns a dict of ${s.heads.length} tensor${s.heads.length > 1 ? "s" : ""}`;
  $("copy").disabled = !!errors.length;
  const order = { error: 0, warn: 1, tip: 2 };
  const seen = new Set();
  const issues = D.issues.filter((i) => !seen.has(i.text) && seen.add(i.text)).sort((a, b) => order[a.level] - order[b.level]);
  $("issues-panel").hidden = !issues.length;
  $("issues").innerHTML = issues.map((i) => `<div class="issue ${i.level}"><span class="tag">${{ error: "Error", warn: "Note", tip: "Tip" }[i.level]}</span><span>${esc(i.text)}</span></div>`).join("");
}

// ---------- diagram ----------

const BAR_H = 11, BAR_GAP = 6, LEVEL = BAR_H + BAR_GAP;

function barWidth(stride) {   // wider = finer grid
  return 22 + 15 * Math.max(0, 5 - Math.log2(stride));
}

function curve(x1, y1, x2, y2, cls = "edge", extra = "") {
  const dx = Math.max(24, (x2 - x1) * 0.5);
  return `<path class="${cls}" d="M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}"${extra}/>`;
}

function curveYAt(x1, y1, x2, y2, x) {   // y where a curve() edge crosses x (its x is monotonic in t)
  const dx = Math.max(24, (x2 - x1) * 0.5);
  const bez = (t, a, b, c, d) => (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t * t * c + t ** 3 * d;
  let lo = 0, hi = 1;
  for (let i = 0; i < 30; i++) { const t = (lo + hi) / 2; if (bez(t, x1, x1 + dx, x2 - dx, x2) < x) lo = t; else hi = t; }
  return bez(lo, y1, y1, y2, y2);
}

function decoderStages(type, spec, info, width) {
  // bars from the output (finest) down to the deepest stage the decoder builds
  if (spec.error) return [];
  if (["unet", "unetplusplus", "fastai_unet"].includes(type)) {
    const deepest = Math.max(...info.strides);
    const stages = [];
    const widths = width && width !== "full" ? DATA.unet_widths[width] : DEFAULT_UNET;
    for (let st = deepest / 2, k = 0; st >= spec.stride; st /= 2, k++) stages.push({ stride: st, ch: type === "unet" ? widths[k] : null });
    stages.reverse();
    stages[0].ch = spec.channels;
    return stages;
  }
  return [{ stride: spec.stride, ch: spec.channels }];
}

function headGlyph(h, x, y, color) {
  const S = 40;
  let g = `<rect x="${x}" y="${y}" width="${S}" height="${S}" rx="7" fill="var(--surface)" stroke="var(--line-strong)"/>`;
  if (h.scope === "pixel") {
    const n = 4, c = (S - 10) / n;
    const pal = ["var(--in1)", "var(--in2)", "var(--in3)", "var(--in4)", "var(--line-strong)"];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let fill, op = 1;
      if (h.type === "regression") { fill = color; op = 0.15 + 0.85 * (((i * 7 + j * 3) % 5) / 4); }
      else { const k = (i + j + (i * j) % 3) % Math.min(5, Math.max(2, h.channels + 1)); fill = h.channels === 1 ? (k ? color : "var(--line)") : pal[k]; }
      g += `<rect x="${x + 5 + j * c + 0.6}" y="${y + 5 + i * c + 0.6}" width="${c - 1.2}" height="${c - 1.2}" rx="1.5" fill="${fill}" opacity="${op.toFixed(2)}"/>`;
    }
  } else if (h.type === "regression") {
    g += `<text x="${x + S / 2}" y="${y + S / 2 + 4}" text-anchor="middle" style="font: 600 12px var(--f-mono); fill: var(--ink)">${h.uncertainty ? "\u03bc,\u03c3\u00b2" : "0.37"}</text>`;
  } else {
    [0.85, 0.45, 0.25].forEach((w, i) => { g += `<rect x="${x + 6}" y="${y + 8 + i * 9}" width="${(S - 12) * w}" height="6" rx="2" fill="${color}" opacity="${1 - i * 0.25}"/>`; });
  }
  return g;
}

function drawDiagram(s, D) {
  const P = [];  // svg parts (drawn in order: edges first, then nodes)
  const E = [];
  const ftype = D.ftype;
  const rasters = s.inputs;
  const cols = [];
  cols.push({ key: "in", title: "INPUTS", w: 110 });
  if (ftype === "early") cols.push({ key: "stack", title: "STACK", w: 70 });
  cols.push({ key: "enc", title: "ENCODERS", w: 150 });
  if (ftype === "intermediate") cols.push({ key: "fuse", title: "FUSION", w: 104 });
  const hasDec = D.needsShared || Object.keys(D.task).length;
  if (hasDec || D.imageHeads.length) cols.push({ key: "dec", title: hasDec ? "DECODER" : "POOL", w: 130 });
  if (ftype === "late") cols.push({ key: "fuse2", title: "FUSION", w: 104 });
  cols.push({ key: "heads", title: "HEADS", w: 250 });
  const GAP = 64;
  let x = 16;
  const X = {};
  for (const c of cols) { X[c.key] = x; c.x = x; x += c.w + GAP; }
  const W = x - GAP + 10;
  const TOP = 44;

  // decoders to draw: the shared one first, then one per head with its own decoder
  const decList = [];
  if (D.needsShared) decList.push({ key: "", type: s.decoder.type, spec: D.shared, width: s.decoder.width });
  for (const [hn, t] of Object.entries(D.task)) decList.push({ key: hn, type: t.type, spec: t, width: t.width || "full", input: t.input || "", sharesWith: t.sharesWith || "" });
  const heights = decList.map((dd) => Math.max(1, decoderStages(dd.type, dd.spec, D.primary, dd.width).length) * LEVEL + 38);
  // late fusion builds every decoder once per input: the first sits level with the row, the rest below it
  const lateExtra = ftype === "late" ? heights.slice(1).reduce((a, h) => a + h + 14, 0) : 0;

  // rows: one per raster input, sized by its encoder's levels (and its decoders, for late fusion)
  const encs = D.encoders;
  const rows = [];
  let y = TOP;
  // decoder stages finer than the encoder's finest level stack above the encoder bars: make room
  const mainSpec = decList.length ? decList[0].spec : null;
  const riseAbove = (info) => (mainSpec && !mainSpec.error && UNET_FAMILY.includes(decList[0].type))
    ? Math.max(0, Math.log2(Math.min(...info.strides) / mainSpec.stride)) * LEVEL : 0;
  rasters.forEach((r, i) => {
    const enc = ftype === "early" ? encs[0] : encs[i];
    const levels = ftype === "early" ? 0 : enc.info.strides.length;
    if ((i === 0 && ftype !== "intermediate") || ftype === "late") y += Math.max(0, riseAbove(enc.info) + 10);
    const base = Math.max(92, levels * LEVEL + 34);
    const h = lateExtra ? Math.max(base, (base - 34) / 2 + 6 + (heights[0] - 38) / 2 + 38 + 14 + lateExtra) : base;
    rows.push({ y, h, cy: y + (base - 34) / 2 + 6 });
    y += h + 22;
  });
  const rasterBottom = y - 22;
  let metaRow = null;
  if (s.meta) {
    const h = Math.max(60, s.meta.fields.length * 24 + 30);
    metaRow = { y: y + 6, h };
    y += h + 28;
  }

  // ---- inputs ----
  rasters.forEach((r, i) => {
    const R = rows[i], col = INPUT_COLORS[i % 4];
    const sz = Math.min(64, Math.max(14, 50 * Math.sqrt(r.res))), cx = X.in + 50, top = R.cy - sz / 2;
    const isTs = r.kind === "timeseries";
    const layers = isTs ? Math.min(6, Math.max(2, r.dates)) : Math.min(4, Math.max(1, Math.ceil(r.channels / 3)));
    for (let k = layers - 1; k >= 0; k--) {
      P.push(`<rect x="${cx - sz / 2 + k * 5}" y="${top - k * 5}" width="${sz}" height="${sz}" rx="4" fill="${col}" opacity="${k ? (isTs ? 0.85 - k * 0.12 : 0.22) : 1}" stroke="var(--surface)" stroke-width="1.5"/>`);
    }
    R.inX = cx + sz / 2 + (layers - 1) * 5 + 6;
    R.inL = cx - sz / 2;
    R.name = r.name;
    P.push(`<text class="nm" x="${cx}" y="${R.cy + 46}" text-anchor="middle">${esc(r.name)}</text>`);
    P.push(`<text class="sub" x="${cx}" y="${R.cy + 60}" text-anchor="middle">${r.channels} ch${isTs ? ` \u00d7 ${r.dates} dates` : ` \u00b7 ${i === 0 ? "primary grid" : r.alignment === "same_grid" ? "same grid" : resLabel(r.res)}`}</text>`);
    if (isTs) P.push(`<text class="sub" x="${cx}" y="${R.cy + 73}" text-anchor="middle">${i === 0 ? "primary grid" : r.alignment === "same_grid" ? "same grid" : resLabel(r.res)}</text>`);
  });

  // ---- encoders ----
  const encPorts = [];   // per encoder: [{stride, x, y, ch}]
  const encNodes = ftype === "early" ? [{ enc: encs[0], cy: (rows[0].cy + rows[rows.length - 1].cy) / 2, color: "var(--ink-2)" }]
    : encs.map((e, i) => ({ enc: e, cy: rows[i].cy, color: INPUT_COLORS[i % 4] }));
  if (ftype === "early") {
    const sx = X.stack + 20, sy = encNodes[0].cy;
    P.push(`<rect x="${sx}" y="${sy - 22}" width="34" height="44" rx="6" class="box"/>`);
    P.push(`<text x="${sx + 17}" y="${sy + 4}" text-anchor="middle" style="font: 600 12px var(--f-mono)">${encs[0].channels}</text>`);
    P.push(`<text class="sub" x="${sx + 17}" y="${sy + 38}" text-anchor="middle">channels</text>`);
    rows.forEach((R) => E.push(curve(R.inX, R.cy, sx, sy)));
    E.push(`<line class="edge" x1="${sx + 34}" y1="${sy}" x2="${X.enc - 4}" y2="${sy}" marker-end="url(#arr)"/>`);
  }
  encNodes.forEach(({ enc, cy, color }, i) => {
    const info = enc.info, n = info.strides.length;
    const top = cy - (n * LEVEL - BAR_GAP) / 2;
    const bx = X.enc + 52;
    const ports = [];
    const railX = bx + barWidth(2) + 8;   // every edge leaves from here, so the curves fan out evenly
    const primStrides = encNodes[0].enc.info.strides;
    // a level ManySeg never reads (no decoder or fusion op takes it: UPerNet, FPN and SegFormer skip /2,
    // DeepLabV3+ reads /4 and the deepest) is dimmed, with no edge
    const lv = D.readLevels ? D.readLevels.get(enc.input) : null;
    info.strides.forEach((st, k) => {
      const w = barWidth(st), yy = top + k * LEVEL;
      // a level of a secondary input that no fusion op reads is drawn dimmed, with no edge
      const unused = lv ? !lv.has(st) : ftype === "intermediate" && i > 0 && !(primStrides.includes(st) && D.fused.includes(st));
      const unfused = unused && ftype === "intermediate" && i > 0;
      P.push(`<rect x="${bx}" y="${yy}" width="${w}" height="${BAR_H}" rx="3" fill="${color}" opacity="${unused ? 0.25 : enc.shared ? 0.55 : 1}"/>`);
      P.push(`<text class="mono" x="${bx - 6}" y="${yy + 8.5}" text-anchor="end"${unused ? ' opacity="0.5"' : ""}>${info.channels[k]}\u00b7/${st}</text>`);
      if (unused) P.push(`<text class="mono" x="${bx + w + 6}" y="${yy + 8.5}" opacity="0.7">${unfused ? "not fused" : "not read"}</text>`);
      if (!unused) E.push(`<line class="edge" x1="${bx + w}" y1="${yy + BAR_H / 2}" x2="${railX}" y2="${yy + BAR_H / 2}" style="stroke:${color};opacity:.75"/>`);
      ports.push({ stride: st, ch: info.channels[k], x: railX - 3, y: yy + BAR_H / 2, unused });
    });
    encPorts.push(ports);
    if (ftype !== "early") E.push(`<line class="edge" x1="${rows[i].inX}" y1="${cy}" x2="${X.enc - 4}" y2="${cy}" marker-end="url(#arr)"/>`);
    const label = enc.shared ? `shares ${enc.shared}${enc.inputs[0].share_norm ? " + BN" : ""}` : info.name;
    P.push(`<text class="sub" x="${bx + 30}" y="${top + n * LEVEL + 12}" text-anchor="middle" style="font-family: var(--f-mono); font-size: 10.5px">${esc(label.length > 24 ? label.slice(0, 23) + "\u2026" : label)}</text>`);
    const r0 = enc.inputs.length === 1 ? enc.inputs[0] : null;
    if (r0 && r0.kind === "timeseries")
      P.push(`<text class="sub" x="${bx + 30}" y="${top + n * LEVEL + 25}" text-anchor="middle" style="fill: var(--accent-ink)">per date, then ${esc(r0.temporal)}</text>`);
  });

  // ---- what feeds the decoder(s) ----
  let feed = encPorts[0];   // ports {stride, x, y}
  const fuseX = X.fuse;
  if (ftype === "intermediate") {
    const prim = encPorts[0];
    const n = prim.length, cy = (rows[0].cy + rows[rows.length - 1].cy) / 2;
    const bh = Math.max(84, n * 20 + 34), top = cy - bh / 2;
    P.push(`<rect x="${fuseX}" y="${top}" width="104" height="${bh}" rx="10" class="box"/>`);
    P.push(`<text class="nm" x="${fuseX + 52}" y="${cy - 2}" text-anchor="middle">${esc(s.fusion.op.replace("_", " "))}</text>`);
    P.push(`<text class="sub" x="${fuseX + 52}" y="${cy + 13}" text-anchor="middle">${D.fused.length ? "at /" + D.fused.join(", /") : "no shared strides"}</text>`);
    const portY = (k) => top + 17 + (k + 0.5) * ((bh - 34) / n);
    feed = prim.map((p, k) => ({ stride: p.stride, x: fuseX + 104, y: portY(k) }));
    encPorts.forEach((ports, i) => ports.forEach((p) => {
      const k = prim.findIndex((q) => q.stride === p.stride);
      if (k < 0 || p.unused) return;
      const fusedHere = D.fused.includes(p.stride);
      if (i > 0 && !fusedHere) return;
      E.push(curve(p.x + 3, p.y, fuseX, portY(k), `edge${i > 0 || fusedHere ? "" : " faint"}`, ` style="stroke:${INPUT_COLORS[i % 4]};opacity:${fusedHere ? 0.75 : 0.35}"`));
    }));
  }

  // ---- decoders ----
  const decNodes = [];   // {key, type, spec, cy, outX, outY, ports}
  const decX = X.dec;
  const centerY = (rows[0].cy + rows[rows.length - 1].cy) / 2;
  if (ftype === "late") {
    rows.forEach((R, i) => {
      let yy = R.cy + (heights[0] - 38) / 2 + 38 + 14;
      decList.forEach((dd, k) => {
        if (k === 0) { decNodes.push({ ...dd, cy: R.cy, row: i, entry: 0 }); return; }
        decNodes.push({ ...dd, cy: yy + (heights[k] - 38) / 2, row: i, entry: k }); yy += heights[k] + 14;
      });
    });
  } else {
    // the main decoder sits level with the features it reads (straight edges); others stack below
    let yy = centerY + (heights[0] - 38) / 2 + 38 + 14;
    decList.forEach((dd, k) => {
      if (k === 0) { decNodes.push({ ...dd, cy: centerY }); return; }
      decNodes.push({ ...dd, cy: yy + (heights[k] - 38) / 2 }); yy += heights[k] + 14;
    });
  }
  let decBottom = TOP;
  const labelBottom = new Map();   // stack (late fusion: row) -> bottom of the last decoder's label
  decNodes.forEach((dn, k) => {
    const stages = decoderStages(dn.type, dn.spec, D.primary, dn.width);
    const n = Math.max(1, stages.length);
    const bx = decX + 24;
    // the main decoder's label sits below its lowest incoming edge, which can reach past the slot the
    // layout gave it: push the decoders stacked under it down to clear that label
    const grp = ftype === "late" ? dn.row : 0;
    const isMain = ftype === "late" ? dn.entry === 0 : k === 0;
    if (!isMain && labelBottom.has(grp)) dn.cy += Math.max(0, labelBottom.get(grp) + 12 - (dn.cy - (n * LEVEL - BAR_GAP) / 2));
    // per-input heads' decoders read their own input's (unfused) encoder levels
    const ownIdx = dn.input ? rasters.findIndex((r) => r.name === dn.input) : -1;
    const src = ftype === "late" ? encPorts[dn.row] : ownIdx >= 0 && encPorts[ownIdx] ? encPorts[ownIdx] : feed;
    const main = ftype === "late" ? dn.entry === 0 : k === 0;
    // the main decoder's stages sit level with the encoder feature of the same stride (the skip each
    // stage fuses), so those edges run straight; finer stages continue upward at the same spacing
    let barY;   // stride -> bar centre y
    if (main && src.length) {
      const sp = src.length > 1 ? (src[1].y - src[0].y) / Math.log2(src[1].stride / src[0].stride) : LEVEL;
      barY = (st) => { const q = src.find((p) => p.stride === st); return q ? q.y : src[0].y - sp * Math.log2(src[0].stride / st); };
    } else {
      const top0 = dn.cy - (n * LEVEL - BAR_GAP) / 2;
      barY = (st) => top0 + Math.max(0, stages.findIndex((x) => x.stride === st)) * LEVEL + BAR_H / 2;
    }
    const ys = stages.map((st) => barY(st.stride));
    const top = (ys.length ? Math.min(...ys) : dn.cy) - BAR_H / 2;
    const bottom = (ys.length ? Math.max(...ys) : dn.cy) + BAR_H / 2;
    if (dn.spec.error) {
      P.push(`<rect x="${bx}" y="${top - 4}" width="96" height="22" rx="5" class="err-box"/>`);
      P.push(`<text class="sub" x="${bx + 48}" y="${top + 11}" text-anchor="middle" style="fill:var(--err)">not buildable</text>`);
    }
    stages.forEach((st, j) => {
      const w = barWidth(st.stride), yy = ys[j] - BAR_H / 2;
      P.push(`<rect x="${bx}" y="${yy}" width="${w}" height="${BAR_H}" rx="3" fill="var(--dec)" opacity="${(1 - j * 0.15).toFixed(2)}"/>`);
      if (j === 0) dn.outX = bx + w, dn.outY = ys[j];
    });
    if (!stages.length) { dn.outX = bx + 96; dn.outY = top + 7; }
    // long labels stop before the heads column (about 23 monospace characters fit)
    const full = dn.key ? `${dn.type} \u00b7 ${dn.key}${dn.sharesWith ? ` = ${dn.sharesWith}` : ""}` : dn.type;
    const label = full.length > 23 ? full.slice(0, 22) + "\u2026" : full;
    const out = dn.spec.error ? "" : `\u2192 ${dn.spec.channels} ch at /${dn.spec.stride}`;
    // label below the bars and below the lowest feature edge coming in, so no edge crosses it
    const lowestIn = main && src.length ? Math.max(...src.map((p) => p.y)) + 4 : bottom;
    const ly = Math.max(bottom, lowestIn, dn.spec.error ? top + 18 : bottom) + 13;
    P.push(`<text class="nm halo" x="${bx}" y="${ly}" style="font-family: var(--f-mono); font-size: 11.5px">${esc(label)}</text>`);
    P.push(`<text class="mono halo" x="${bx}" y="${ly + 13}">${out}</text>`);
    decBottom = Math.max(decBottom, ly + 18);
    labelBottom.set(grp, ly + 18);
    // edges: each feature goes to the stage of its stride; deeper features feed the deepest stage
    const deepest = stages.length ? stages[stages.length - 1].stride : 0;
    dn.portY = src.map((p) => {
      if (!main || !stages.length) {
        const ph = Math.max(n * LEVEL, 30), py0 = dn.cy - ph / 2, j = src.indexOf(p);
        return py0 + (j + 0.5) * (ph / src.length);
      }
      const st = stages.some((x) => x.stride === p.stride) ? p.stride : (p.stride > deepest ? deepest : stages[0].stride);
      return barY(st);
    });
    // only the levels this decoder reads (UPerNet, FPN, SegFormer skip /2; DeepLabV3+ reads /4 and the deepest)
    let reads = dn.spec.read && dn.spec.read.length ? dn.spec.read : null;
    if (ftype === "late") {   // every input runs the decoder on its own encoder's levels
      const d0 = s.decoder, own = !dn.key;
      const key = FIXED_DECODERS.includes(dn.type) ? dn.type : dn.type === "unet" ? (own ? `unet/${d0.out_stride}/${d0.width}` : "unet/1/full") : `${dn.type}/${own ? d0.out_stride : 1}`;
      const v = encs[dn.row].info.dec[key];
      reads = Array.isArray(v) && v[3] && v[3].length ? v[3] : null;
    }
    dn.inEdges = [];
    src.forEach((p, j) => {
      if (reads && !reads.includes(p.stride)) return;
      dn.inEdges.push({ x1: p.x + 3, y1: p.y, x2: bx - 4, y2: dn.portY[j] });
      const col = ftype === "late" ? INPUT_COLORS[dn.row % 4] : ownIdx >= 0 ? INPUT_COLORS[ownIdx % 4] : ftype === "intermediate" ? "var(--edge)" : (ftype === "early" ? "var(--ink-2)" : INPUT_COLORS[0]);
      E.push(curve(p.x + 3, p.y, bx - 4, dn.portY[j], "edge", ` style="stroke:${col};opacity:${main ? 0.7 : 0.35}"`));
    });
    dn.inX = bx - 4;
  });

  // pool node for image heads
  let pool = null;
  if (D.imageHeads.length) {
    const deep = (ftype === "intermediate" ? feed : encPorts[0]);
    const dp = deep[deep.length - 1];
    const py = decNodes.length ? decBottom + 26 : centerY;
    pool = { x: decX + 24 + 14, y: py };
    P.push(`<circle cx="${pool.x}" cy="${pool.y}" r="13" fill="var(--surface)" stroke="var(--line-strong)"/>`);
    P.push(`<text x="${pool.x}" y="${pool.y + 3.5}" text-anchor="middle" style="font: 600 9px var(--f-mono); fill: var(--muted)">GAP</text>`);
    P.push(`<text class="sub halo" x="${pool.x}" y="${pool.y + 27}" text-anchor="middle">global pool of /${dp.stride}</text>`);
    E.push(curve(dp.x + 3, dp.y, pool.x - 13, pool.y, "edge", ` style="stroke-dasharray:3 3"`));
    decBottom = Math.max(decBottom, pool.y + 32);
  }

  // late fusion: one fusion box per decoder (shared, then each head's own), merging its per-input copies
  const late = {};   // key -> {x, y}
  if (ftype === "late" && decNodes.length) {
    const fx = X.fuse2, bh = 64;
    let prevY = -Infinity;
    decList.forEach((dd, k) => {
      const mine = decNodes.filter((dn) => dn.entry === k);
      const cy = Math.max(mine.reduce((a, dn) => a + dn.outY, 0) / mine.length, prevY + bh + 16);
      prevY = cy;
      P.push(`<rect x="${fx}" y="${cy - bh / 2}" width="104" height="${bh}" rx="10" class="box"/>`);
      P.push(`<text class="nm" x="${fx + 52}" y="${cy - 2}" text-anchor="middle">${esc(s.fusion.op.replace("_", " "))}</text>`);
      P.push(`<text class="sub" x="${fx + 52}" y="${cy + 13}" text-anchor="middle">${dd.key ? esc(dd.key) + " decoders" : "decoder outputs"}</text>`);
      mine.forEach((dn) => E.push(curve(dn.outX + 3, dn.outY, fx, cy - 10 + dn.row * (20 / Math.max(1, mine.length - 1)), "edge", ` style="stroke:${INPUT_COLORS[dn.row % 4]}"`)));
      late[dd.key] = { x: fx + 104, y: cy };
      decBottom = Math.max(decBottom, cy + bh / 2 + 10);
    });
  }

  // ---- heads ----
  // each head is placed level with its source where it can be (straight edge), in order, never overlapping
  const HH = 62;
  const hx = X.heads;
  const shapes = outputShapes(s, D);
  const srcs = s.heads.map((h) => {
    if (h.scope === "image") return pool ? { x: pool.x + 13, y: pool.y } : null;
    if (ftype === "late") return late[h.decoder ? h.name : ""] || null;
    const dn = decNodes.find((n) => n.key === (h.decoder || h.input ? h.name : "")) || decNodes[0];
    return dn ? { x: dn.outX + 3, y: dn.outY } : null;
  });
  // stack in the order of their sources' heights (stable), so edges never cross
  const order = s.heads.map((_, i) => i).sort((a, b) => (srcs[a] ? srcs[a].y : 0) - (srcs[b] ? srcs[b].y : 0) || a - b);
  const gys = new Array(s.heads.length);
  let prev = -Infinity;
  for (const i of order) {
    const want = srcs[i] ? srcs[i].y - 20 : TOP;
    gys[i] = Math.max(TOP, want, prev + HH);
    prev = gys[i];
  }
  s.heads.forEach((h, i) => {
    const gy = gys[i];
    P.push(headGlyph(h, hx, gy, h.type === "regression" ? "var(--in4)" : "var(--in1)"));
    P.push(`<text class="nm" x="${hx + 50}" y="${gy + 12}">${esc(h.name)}</text>`);
    const nh = h.scope === "pixel" ? headOptions(h).hidden.length : 0;
    const desc = `${h.type} \u00b7 ${h.channels} ${h.type === "regression" ? "val" : "cls"}${h.uncertainty ? " \u00b1\u03c3" : ""}${h.activation ? " \u2265 0" : ""} \u00b7 ${h.scope}${h.scale !== 1 ? ` \u00b7 \u00d7${h.scale}` : ""}${nh ? ` \u00b7 +${nh} conv` : ""}`;
    P.push(`<text class="sub" x="${hx + 50}" y="${gy + 26}">${esc(desc)}</text>`);
    P.push(`<text class="shape" x="${hx + 50}" y="${gy + 40}">(${shapes[i].shape.join(", ")})</text>`);
    if (srcs[i]) E.push(curve(srcs[i].x, srcs[i].y, hx - 4, gy + 20, "edge", ` marker-end="url(#arr)"`));
  });
  const headsBottom = gys.length ? Math.max(...gys) + HH - 8 : TOP;

  // ---- metadata ----
  if (s.meta && metaRow) {
    const m = s.meta;
    const fy0 = metaRow.y + 4;
    m.fields.forEach((f, j) => {
      const fy = fy0 + j * 24, label = f.name.length > 12 ? f.name.slice(0, 11) + "\u2026" : f.name;
      P.push(`<rect x="${X.in + 8}" y="${fy}" width="84" height="18" rx="9" fill="var(--meta-soft)" stroke="var(--meta)"/>`);
      P.push(`<text x="${X.in + 50}" y="${fy + 12.5}" text-anchor="middle" style="font: 500 10.5px var(--f-ui); fill: var(--meta-ink)">${esc(label)}</text>`);
    });
    const midY = fy0 + (m.fields.length * 24 - 6) / 2;
    P.push(`<text class="nm" x="${X.in + 50}" y="${fy0 + m.fields.length * 24 + 12}" text-anchor="middle">${esc(m.name)}</text>`);
    const mx = X.enc + 52;
    E.push(`<line class="edge" x1="${X.in + 96}" y1="${midY}" x2="${mx - 6}" y2="${midY}" marker-end="url(#arr)"/>`);
    P.push(`<rect x="${mx}" y="${midY - 12}" width="54" height="24" rx="12" fill="var(--meta)"/>`);
    P.push(`<text x="${mx + 27}" y="${midY + 4}" text-anchor="middle" style="font: 700 11px var(--f-ui); fill: var(--surface)">MLP</text>`);
    P.push(`<text class="sub" x="${mx + 27}" y="${midY + 28}" text-anchor="middle">embedding</text>`);
    const ex = mx + 54;
    if (m.mode === "film" && decNodes.length) {
      // a vertical amber rail with a FiLM square on each feature edge a decoder reads. U-Net edges run
      // straight into their stages, so the rail sits by the decoder; other decoders merge their inputs
      // into one port, so the rail sits near the feature ports, where those edges are still apart
      const edges = decNodes.flatMap((dn) => dn.inEdges);
      const straight = decNodes.every((dn) => UNET_FAMILY.includes(dn.type));
      const railX = straight || !edges.length ? X.dec - 18 : Math.min(X.dec - 18, Math.max(...edges.map((e) => e.x1)) + 42);
      const targets = [];
      for (const y of edges.map((e) => curveYAt(e.x1, e.y1, e.x2, e.y2, railX)).sort((a, b) => a - b))
        if (!targets.length || y - targets[targets.length - 1] > 3) targets.push(y);
      const topY = Math.min(...targets);
      E.push(`<path class="meta-edge" d="M${ex},${midY} L${railX - 8},${midY} Q${railX},${midY} ${railX},${midY - 8} L${railX},${topY}"/>`);
      for (const ty of targets) P.push(`<rect x="${railX - 4}" y="${ty - 4}" width="8" height="8" rx="1.5" fill="var(--meta)"/>`);
      // by the encoder the space above the rail can hold a dimmed level's label: name the rail beside its riser
      const botY = Math.max(...targets);
      if (straight || midY - botY < 30) P.push(`<text x="${railX}" y="${topY - 10}" text-anchor="middle" style="font: 700 10px var(--f-ui); fill: var(--meta)">FiLM</text>`);
      else P.push(`<text x="${railX + 7}" y="${(botY + midY) / 2 + 4}" style="font: 700 10px var(--f-ui); fill: var(--meta)">FiLM</text>`);
      if (pool) P.push(`<rect x="${pool.x - 4 - 16}" y="${pool.y - 4}" width="8" height="8" rx="1.5" fill="var(--meta)"/>`);
    } else if (m.mode === "film" || m.mode === "head") {
      const anyImage = s.heads.some((h) => h.scope === "image");
      const tgt = s.heads.map((h, i) => ({ h, i })).filter(({ h }) => m.mode === "film" || h.scope === "image" || !anyImage);
      for (const { i } of tgt) E.push(curve(ex, midY, hx - 4, gys[i] + 30, "meta-edge", ` style="stroke-dasharray:4 3"`));
    } else if (m.mode === "channels") {
      const R = rows[0];
      if (rows.length === 1) {
        E.push(`<path class="meta-edge" d="M${X.in + 50},${fy0 - 4} L${X.in + 50},${R.cy + 68}" stroke-dasharray="4 3"/>`);
        P.push(`<text class="sub" x="${X.in + 56}" y="${(fy0 + R.cy + 64) / 2}" style="fill:var(--meta)">as channels</text>`);
      } else {   // the primary input is at the top: go up the left margin, clear of the other inputs
        const lx = X.in - 8, boxL = R.inL - 4;
        E.push(`<path class="meta-edge" d="M${X.in + 4},${fy0 + 9} L${lx},${fy0 + 9} L${lx},${R.cy} L${boxL},${R.cy}" stroke-dasharray="4 3" marker-end="url(#arr)"/>`);
        P.push(`<text class="sub" x="${X.in + 50}" y="${fy0 + m.fields.length * 24 + 26}" text-anchor="middle" style="fill:var(--meta)">as channels of ${esc(R.name)}</text>`);
      }
    }
  }

  const H = Math.max(y, decBottom + 16, headsBottom + 20, TOP + 120);
  const titles = cols.map((c) => `<text class="colt" x="${c.key === "heads" ? c.x + 4 : c.x + c.w / 2}" y="22" text-anchor="${c.key === "heads" ? "start" : "middle"}">${c.title}</text>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Diagram of the configured model: inputs, encoders, fusion, decoders and heads">
    <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 1 L9 5 L0 9 z" fill="var(--edge)"/></marker></defs>
    ${titles}<line x1="12" y1="32" x2="${W - 12}" y2="32" stroke="var(--line)"/>
    <g>${E.join("")}</g><g>${P.join("")}</g></svg>`;
}

// ---------- tooltips ----------
// hover or focus shows a tip; clicking an (i) pins it open (touch), Escape or an outside click closes it

const tip = document.createElement("div");
tip.id = "tip";
tip.setAttribute("role", "tooltip");
tip.hidden = true;
document.body.appendChild(tip);
let tipFor = null, pinned = false;

function showTip(el) {
  const html = TIPS[el.dataset.tipk];
  if (!html) return;
  tip.innerHTML = html;
  tip.hidden = false;
  tipFor = el;
  const r = el.getBoundingClientRect(), t = tip.getBoundingClientRect();
  const vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight;
  let x = Math.min(Math.max(8, r.left + r.width / 2 - t.width / 2), vw - t.width - 8);
  let y = r.bottom + 8;
  if (y + t.height > vh - 8) y = Math.max(8, r.top - t.height - 8);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
function hideTip() { tip.hidden = true; tipFor = null; pinned = false; }

document.addEventListener("mouseover", (e) => {
  const el = e.target.closest && e.target.closest("[data-tipk]");
  if (pinned) return;
  if (el && el !== tipFor) showTip(el);
  else if (!el && tipFor) hideTip();
});
document.addEventListener("focusin", (e) => {
  const el = e.target.closest && e.target.closest("[data-tipk]");
  if (el) showTip(el); else if (!pinned) hideTip();
});
document.addEventListener("click", (e) => {
  const b = e.target.closest && e.target.closest(".info");
  if (b) { e.preventDefault(); if (pinned && tipFor === b) hideTip(); else { showTip(b); pinned = true; } return; }
  if (pinned) hideTip();
}, true);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTip(); });
window.addEventListener("scroll", () => { if (!pinned) hideTip(); else if (tipFor && document.contains(tipFor)) showTip(tipFor); }, { passive: true });

renderExamples();
renderForm();
renderPreview();
