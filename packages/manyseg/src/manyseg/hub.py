"""Share models on the Hugging Face Hub.

``push_to_hub(model, "org/name", ...)`` saves the model with :func:`manyseg.save` (same
arguments: normalisation, activations, classes, bands) as ``model.safetensors`` in a Hub model
repository, with a generated model card listing the inputs, heads and the licences of the
pretrained weights the model started from. ``manyseg.load("hf-hub:org/name")`` downloads and
loads it (``"hf-hub:org/name@revision"`` for a branch, tag or commit), as timm names Hub models;
:func:`manyseg.read_metadata` takes the same names. Downloads are cached by ``huggingface_hub``
(a timm dependency), and ``token=``, ``HF_TOKEN`` or a stored login (``hf auth login``;
``huggingface-cli login`` before huggingface_hub 0.34) give access to private repositories.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import DenseModel

PREFIX = "hf-hub:"
FILENAME = "model.safetensors"


def split_name(name: str) -> tuple[str, str | None]:
    """``"hf-hub:org/name@rev"`` -> ``("org/name", "rev")`` (revision None when not given)."""
    repo = name.removeprefix(PREFIX)
    repo, _, revision = repo.partition("@")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(f"{name!r}: expected 'hf-hub:<org>/<name>' or "
                         f"'hf-hub:<org>/<name>@<revision>'.")
    return repo, revision or None


def download(name: str, *, token: str | None = None, cache_dir: str | Path | None = None) -> Path:
    """The local path of ``model.safetensors`` from ``"hf-hub:org/name[@revision]"``, downloaded
    on first use and cached."""
    from huggingface_hub import hf_hub_download

    repo, revision = split_name(name)
    return Path(hf_hub_download(repo, FILENAME, revision=revision, token=token,
                                cache_dir=cache_dir, library_name="manyseg"))


def push_to_hub(model: DenseModel, repo_id: str, *, normalize: Any = None,
                activations: Mapping[str, str] | None = None,
                classes: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
                bands: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
                extra: Mapping[str, Any] | None = None, license: str | None = None,
                private: bool = True, model_card: bool | str = True,
                revision: str | None = None, commit_message: str = "Upload ManySeg model",
                token: str | None = None) -> str:
    """Save ``model`` with :func:`manyseg.save` and upload it to the Hub model repository
    ``repo_id`` (``"org/name"``), creating the repository if needed. Returns the commit URL.

    Args:
        normalize, activations, classes, bands, extra: as for :func:`manyseg.save`.
        license: The licence of your fine-tuned weights, as a Hub licence id (``"mit"``,
            ``"cc-by-4.0"``, ...), for the model card. It must respect the licences of the
            pretrained weights the model started from, which the card lists.
        private: Create a new repository as private (the default); an existing repository
            keeps its visibility. Load a private model with ``token=`` or a stored login.
        model_card: True generates a ``README.md`` model card when the branch has none (an
            existing card, perhaps edited on the Hub, is kept); a string is uploaded as the
            card; False uploads no card.
        revision: The branch to push to, created from the main branch if it does not exist
            (default: the main branch). Load it as ``"hf-hub:org/name@<revision>"``. Tags and
            commits can be loaded but not pushed to.
        token: A Hub token with write access (default: ``HF_TOKEN`` or the stored login).
    """
    from huggingface_hub import HfApi

    from .io import read_metadata, save

    split_name(repo_id)  # validates the name before anything is written
    if revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"revision={revision!r} is a commit; push to a branch name.")
    with tempfile.TemporaryDirectory() as tmp:
        path = save(model, Path(tmp) / FILENAME, normalize=normalize, activations=activations,
                    classes=classes, bands=bands, extra=extra)
        api = HfApi(token=token, library_name="manyseg")
        api.create_repo(repo_id, private=private, exist_ok=True)
        _ensure_branch(api, repo_id, revision)
        if model_card is True and not api.file_exists(repo_id, "README.md", revision=revision):
            model_card = card(read_metadata(path), repo_id, license=license, revision=revision)
        if isinstance(model_card, str):
            (Path(tmp) / "README.md").write_text(model_card, encoding="utf-8")
        info = api.upload_folder(repo_id=repo_id, folder_path=tmp, revision=revision,
                                 commit_message=commit_message)
    return str(getattr(info, "commit_url", info))


def _ensure_branch(api, repo_id: str, revision: str | None) -> None:
    """Make ``revision`` a branch a commit can go to. A commit to a missing branch fails, a
    branch is created from main, and a new repository is empty (main has no commit yet), so
    main first gets a ``.gitattributes`` commit when needed."""
    if revision is None or revision == "main":
        return
    refs = api.list_repo_refs(repo_id)
    branches = {b.name for b in refs.branches}
    if revision in branches:
        return
    if revision in {t.name for t in refs.tags}:
        raise ValueError(f"revision={revision!r} is a tag of {repo_id}; push to a branch name.")
    if "main" not in branches:
        api.upload_file(path_or_fileobj=b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
                        path_in_repo=".gitattributes", repo_id=repo_id,
                        commit_message="Initial commit")
    api.create_branch(repo_id, branch=revision, exist_ok=True)


def _cell(value: Any) -> str:
    """``value`` as one Markdown table cell."""
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _names(value: Any, key: str | None) -> str:
    names = value.get(key) if isinstance(value, Mapping) else value
    return ", ".join(_cell(n) for n in names) if names else "—"


def _heads(meta: Mapping[str, Any]) -> dict[str, dict]:
    """Each saved head's config (type, scope, out_channels), defaults filled in."""
    config = meta["create_model"]
    heads = config.get("heads") or {"logits": {"type": "segmentation",
                                               "out_channels": config.get("num_classes")}}
    out = {}
    for name in meta["heads"]:
        cfg = dict(heads.get(name) or {})
        cfg.setdefault("type", "segmentation")
        cfg.setdefault("scope", "image" if cfg["type"] == "classification" else "pixel")
        if cfg.get("uncertainty") and cfg.get("out_channels"):
            cfg["out_channels"] = f"{2 * cfg['out_channels']} (mean, log-variance)"
        out[name] = cfg
    return out


def card(meta: Mapping[str, Any], repo_id: str, *, license: str | None = None,
         revision: str | None = None) -> str:
    """A Hub model card (``README.md``) for a file's metadata (:func:`manyseg.read_metadata`):
    how to load it, its inputs and normalisation, its heads, and the pretrained weights it
    started from with their licences and citations."""
    config, heads = meta["create_model"], _heads(meta)
    kinds = {(c["type"], c["scope"]) for c in heads.values()}
    pipeline = ("image-segmentation" if ("segmentation", "pixel") in kinds else
                "image-classification" if kinds == {("classification", "image")} else None)
    front = ["---", "library_name: manyseg"]
    if license:
        front.append(f"license: {license}")
    if pipeline:
        front.append(f"pipeline_tag: {pipeline}")
    front += ["tags:", "- manyseg", "- pytorch", "- safetensors", "---", ""]

    name = PREFIX + repo_id + (f"@{revision}" if revision else "")
    lines = [f"# {repo_id.split('/')[-1]}", "",
             f"A [ManySeg](https://github.com/DPIRD-DMA/ManySeg) model, saved by ManySeg "
             f"{meta.get('manyseg_version', 'unknown')} (save format {meta.get('format')}).", "",
             "```python", "import manyseg", "",
             f'model = manyseg.load("{name}")  # returns raw outputs (logits)',
             f'meta = manyseg.read_metadata("{name}")  # normalisation, activations',
             "```", "",
             "The model returns raw outputs: normalise its input as below, and apply each "
             "head's activation to get predictions.", ""]

    inputs = config.get("inputs")
    bands, norm = meta.get("bands"), meta.get("normalize")
    lines += ["## Inputs", ""]
    if inputs is None:
        lines += [f"One tensor `(B, {config.get('in_channels') or 3}, H, W)`; bands: "
                  f"{_names(bands, None)}.", ""]
    else:
        lines += ["The model takes a dict keyed by input name.", "",
                  "| input | kind | shape | bands / fields |", "|---|---|---|---|"]
        notes = []
        for key, cfg in inputs.items():
            kind = cfg.get("kind", "image")
            if kind == "vector":
                fields = cfg.get("fields") or {}
                shape = f"(B, {cfg.get('num_features') or 'N'})"
                desc = (", ".join(f"{_cell(f)} ({_cell(c.get('type', '?'))})"
                                  for f, c in fields.items()) or "raw features")
                notes.append(f"`{_cell(key)}` is built with `model.pack_metadata("
                             f"\"{_cell(key)}\", {{field: values}})`; NaN marks a missing value.")
            else:
                c = cfg.get("in_channels") or 3
                shape = f"(B, T, {c}, H, W)" if kind == "timeseries" else f"(B, {c}, H, W)"
                desc = _names(bands, key)
                if kind == "timeseries":
                    notes.append(f"`{_cell(key)}` also takes optional `{_cell(key)}_mask` "
                                 f"`(B, T)` bool (False for missing dates) and "
                                 f"`{_cell(key)}_dates` `(B, T)` float days.")
            lines.append(f"| `{_cell(key)}` | {kind} | `{shape}` | {desc} |")
        lines += [""] + [f"* {n}" for n in notes] + ([""] if notes else [])
    lines += ["Normalisation (a ManyPred normalisation spec): "
              + (f"`{_json(norm)}`" if norm is not None else "**not stored**: ask the model's "
                 "author."), ""]

    classes = meta.get("classes")
    lines += ["## Outputs", "",
              "One tensor." if meta.get("single_output") or not config.get("heads")
              else "A dict keyed by head name.", "",
              "| head | type | scope | channels | activation | classes |",
              "|---|---|---|---|---|---|"]
    for head, cfg in heads.items():
        lines.append(f"| `{_cell(head)}` | {_cell(cfg['type'])} | {_cell(cfg['scope'])} | "
                     f"{_cell(cfg.get('out_channels') or '—')} | "
                     f"{meta['activations'].get(head, 'none')} | {_names(classes, head)} |")
    lines.append("")

    recs = meta.get("pretrained_weights") or []
    if recs:
        lines += ["## Pretrained weights", "",
                  "The model was fine-tuned from these weights. Their licences still apply, and "
                  "most require attribution:", ""]
        for rec in recs:
            src = f" ([source]({rec['source']}))" if rec.get("source") else ""
            lines.append(f"* `{rec.get('module')}`: {rec.get('weights')}, licence "
                         f"{rec.get('license', 'unknown')}{src}.")
            if rec.get("citation"):
                lines.append(f"  Cite: {rec['citation']}")
        lines.append("")
    return "\n".join(front + lines)


def _json(value: Any) -> str:
    import json

    return json.dumps(value, separators=(", ", ": "))
