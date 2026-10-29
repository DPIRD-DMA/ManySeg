"""manyseg.hub: push_to_hub uploads a manyseg.save file and a model card; load / read_metadata
take "hf-hub:" names. The Hub is replaced by a local folder (no network)."""

import shutil
from pathlib import Path

import manyseg
import pytest
import torch
from manyseg import create_model, hub


class FakeHub:
    """HfApi and hf_hub_download over ``root/<org>/<name>/<branch>``, with the real Hub's rules
    that matter here: a new repository has a main branch, a commit to a missing branch fails,
    and a private repository needs the uploader's token to download."""

    def __init__(self, root: Path, token: str | None = None):
        self.root, self.token, self.created, self.commits = root, token, {}, []

    def repo(self, repo_id, revision=None):
        return self.root / repo_id / (revision or "main")

    # HfApi
    def create_repo(self, repo_id, private=True, exist_ok=False):
        if repo_id not in self.created:
            self.created[repo_id] = private
            self.repo(repo_id).mkdir(parents=True)

    def create_branch(self, repo_id, branch, exist_ok=False):
        if not self.repo(repo_id, branch).exists():
            shutil.copytree(self.repo(repo_id), self.repo(repo_id, branch))

    def file_exists(self, repo_id, filename, revision=None):
        return (self.repo(repo_id, revision) / filename).exists()

    def upload_folder(self, repo_id, folder_path, revision=None, commit_message=""):
        from huggingface_hub.errors import RevisionNotFoundError

        dest = self.repo(repo_id, revision)
        if not dest.exists():
            raise RevisionNotFoundError(f"{repo_id}@{revision}", response=None)
        shutil.copytree(folder_path, dest, dirs_exist_ok=True)
        self.commits.append((repo_id, revision, commit_message, sorted(
            p.name for p in Path(folder_path).iterdir())))
        return f"https://huggingface.co/{repo_id}/commit/{len(self.commits)}"

    # hf_hub_download
    def download(self, repo_id, filename, revision=None, token=None, cache_dir=None,
                 library_name=None):
        if self.created.get(repo_id) and token != self.token:
            raise PermissionError(f"{repo_id} is private")
        path = self.repo(repo_id, revision) / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path)


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    import huggingface_hub

    fake = FakeHub(tmp_path / "hub", token="hf_secret")

    def api(token=None, library_name=None):
        assert token == fake.token, "push_to_hub must pass its token to HfApi"
        return fake

    monkeypatch.setattr(huggingface_hub, "HfApi", api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake.download)
    return fake


def small_model():
    torch.manual_seed(0)
    return create_model(backbone="resnet18", pretrained=False, in_channels=4, heads={
        "water": {"type": "segmentation", "out_channels": 2},
        "depth": {"type": "regression", "out_channels": 1}}).eval()


SAVE = dict(activations={"water": "softmax"}, normalize=[{"type": "scale", "value": 10000}],
            classes={"water": ["dry", "water"]}, bands=["B", "G", "R", "NIR"], token="hf_secret")


def push(repo_id="org/water", model=None, **kw):
    return manyseg.push_to_hub(model or small_model(), repo_id, **{**SAVE, **kw})


def test_push_then_load_roundtrip(fake_hub):
    model = small_model()
    url = push(model=model, license="mit")
    assert url.endswith("/commit/1")
    assert fake_hub.created == {"org/water": True}  # private by default
    assert fake_hub.commits[0][3] == ["README.md", "model.safetensors"]

    loaded = manyseg.load("hf-hub:org/water", token="hf_secret")
    x = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        a, b = model(x), loaded(x)
    assert all(torch.equal(a[k], b[k]) for k in a)
    meta = manyseg.read_metadata("hf-hub:org/water", token="hf_secret")
    assert meta["classes"] == {"water": ["dry", "water"]}
    assert meta["normalize"] == SAVE["normalize"]
    with pytest.raises(PermissionError):  # private: the token is needed
        manyseg.load("hf-hub:org/water")


def test_model_card_contents(fake_hub):
    push(license="cc-by-4.0")
    text = (fake_hub.repo("org/water") / "README.md").read_text()
    assert text.startswith("---\nlibrary_name: manyseg\nlicense: cc-by-4.0\n"
                           "pipeline_tag: image-segmentation\n")
    assert 'manyseg.load("hf-hub:org/water")' in text
    assert "| `water` | segmentation | pixel | 2 | softmax | dry, water |" in text
    assert "| `depth` | regression | pixel | 1 | none | — |" in text
    assert "`(B, 4, H, W)`; bands: B, G, R, NIR" in text and '"value": 10000' in text
    assert "## Pretrained weights" not in text  # pretrained=False


def test_existing_card_is_kept(fake_hub):
    push(model_card="# hand-written\n")
    push()  # a generated card would replace it
    assert (fake_hub.repo("org/water") / "README.md").read_text() == "# hand-written\n"
    assert fake_hub.commits[1][3] == ["model.safetensors"]
    push(model_card=False)
    assert fake_hub.commits[2][3] == ["model.safetensors"]


def test_revision_creates_the_branch(fake_hub):
    push(revision="v1")  # the real Hub refuses a commit to a missing branch
    assert manyseg.read_metadata("hf-hub:org/water@v1", token="hf_secret")["heads"] == [
        "water", "depth"]
    assert 'manyseg.load("hf-hub:org/water@v1")' in (
        fake_hub.repo("org/water", "v1") / "README.md").read_text()
    with pytest.raises(FileNotFoundError):
        manyseg.load("hf-hub:org/water", token="hf_secret")  # nothing on main
    push(revision="v1")  # an existing branch is reused
    assert len(fake_hub.commits) == 2


def test_card_multi_input_vector_timeseries_and_image_heads():
    config = {"inputs": {
        "s2": {"kind": "timeseries", "in_channels": 10},
        "dem": {"in_channels": 1},
        "meta": {"kind": "vector", "fields": {"doy": {"type": "cyclic", "period": 365.25},
                                              "latlon": {"type": "coords"}}}},
        "heads": {"crop": {"type": "segmentation", "out_channels": 3},
                  "yield": {"type": "regression", "out_channels": 1, "scope": "image"},
                  "label": {"type": "classification", "out_channels": 2}}}
    meta = {"create_model": config, "heads": ["crop", "yield", "label"], "format": 2,
            "activations": {"crop": "softmax", "yield": "none", "label": "sigmoid"},
            "classes": {"crop": ["wheat", "canola | lupin", "other"]},
            "bands": {"dem": ["height"]}}
    text = hub.card(meta, "org/crops")
    assert "| `s2` | timeseries | `(B, T, 10, H, W)` | — |" in text
    assert "`s2_mask`" in text and "`s2_dates`" in text
    assert "| `dem` | image | `(B, 1, H, W)` | height |" in text
    assert "| `meta` | vector | `(B, N)` | doy (cyclic), latlon (coords) |" in text
    assert 'model.pack_metadata("meta", {field: values})' in text
    assert "| `crop` | segmentation | pixel | 3 | softmax | wheat, canola \\| lupin, other |" \
        in text
    assert "| `yield` | regression | image | 1 | none | — |" in text
    assert "| `label` | classification | image | 2 | sigmoid | — |" in text
    assert "A dict keyed by head name." in text


def test_card_pipeline_tag_for_classifiers_and_attribution():
    rec = {"module": "encoder", "weights": "ssl4eo_s2_dino", "license": "CC BY 4.0",
           "citation": "Wang et al. 2023", "source": "https://github.com/zhu-xlab/SSL4EO-S12"}
    meta = {"create_model": {"in_channels": 13, "heads": {
                "labels": {"type": "classification", "out_channels": 19}}},
            "heads": ["labels"], "activations": {"labels": "sigmoid"},
            "pretrained_weights": [rec], "format": 2}
    text = hub.card(meta, "org/s2")
    assert "license:" not in text and "pipeline_tag: image-classification" in text
    assert "## Pretrained weights" in text and "Cite: Wang et al. 2023" in text
    assert "One tensor `(B, 13, H, W)`" in text
    assert "**not stored**" in text
    meta["create_model"]["heads"]["labels"]["type"] = "regression"
    assert "pipeline_tag" not in hub.card(meta, "org/s2")


@pytest.mark.parametrize("name", ["hf-hub:water", "hf-hub:a/b/c", "hf-hub:/b", "hf-hub:a/"])
def test_bad_names(name):
    with pytest.raises(ValueError, match="hf-hub:<org>/<name>"):
        hub.split_name(name)


def test_bad_repo_id_fails_before_upload(fake_hub):
    with pytest.raises(ValueError):
        push("water")
    assert fake_hub.created == {}


def test_save_errors_before_creating_repo(fake_hub):
    with pytest.raises(ValueError, match="turn into predictions"):
        push(activations=None)
    assert fake_hub.created == {}


def test_list_classes_or_bands_need_one_head_or_input(tmp_path):
    model = small_model()
    acts = {"activations": {"water": "softmax"}, "normalize": None}
    with pytest.raises(ValueError, match="classes as a list is for a model with one head"):
        manyseg.save(model, tmp_path / "m.safetensors", classes=["dry", "water"], **acts)
    with pytest.raises(ValueError, match="unknown heads"):
        manyseg.save(model, tmp_path / "m.safetensors", classes={"lake": ["a"]}, **acts)
    two = create_model(pretrained=False, num_classes=2, inputs={
        "s2": {"in_channels": 4, "backbone": "resnet18"},
        "s1": {"in_channels": 2, "backbone": "resnet18"}})
    with pytest.raises(ValueError, match="bands as a list is for a model with one input"):
        manyseg.save(two, tmp_path / "m.safetensors", bands=["VV", "VH"],
                     activations={"logits": "softmax"}, normalize=None)
