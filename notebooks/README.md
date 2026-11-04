# Notebooks

Longer walkthroughs with real results and figures. Each one is saved with its outputs, so you can
read it here without running anything. For short reference code on one feature, use the
[scripts in `examples/`](../examples/).

| Notebook | Shows | Data | To run it |
| --- | --- | --- | --- |
| [Mapping floods from 13-band Sentinel-2](01_flood_mapping.ipynb) | A U-Net trained on real images and tested on a flood it hasn't seen, with its best and worst chips | Sen1Floods11, about 400 MB | A few minutes on a GPU |
| [Two sensors in, four outputs out](02_radar_and_multiple_outputs.ipynb) | Optical and radar inputs with gated fusion, a model diagram, and pixel and image-level heads trained together | Sen1Floods11, about 700 MB (300 MB after notebook 1) | A few minutes on a GPU |

The rest cover each vector output, from labels in a real CRS through targets and block-by-block decoding to GeoJSON. The API is experimental. These notebooks use synthetic data and **no trained model**, because the targets stand in for a model's outputs. Each one runs in under a minute on a CPU with nothing to download.

| Notebook | Output | Also tests |
| --- | --- | --- |
| [From a model's outputs to building polygons](03_vector_footprints.ipynb) | Building footprints | How the decoder copes with blurred, noisy outputs |
| [Field boundaries as polygons](04_vector_fields.ipynb) | Touching field parcels | What keeps neighbouring fields from merging |
| [Road networks as graphs](05_vector_roads.ipynb) | Road graphs with overpasses | False junctions from noise, and which gaps get bridged |
| [Roofs with heights](06_vector_roofs.ipynb) | LoD2 roof faces, planes and a 3D mesh | How much height noise the plane fit tolerates |
| [Trees as points and counts](07_vector_points.ipynb) | Points and density counts | How to pick a heatmap threshold, and how counts drift |

## Running them

```bash
pip install "manyseg[vectors]" rasterio matplotlib jupyter
jupyter lab
```

Downloads go to `data/` next to the notebook. Change `DATA` in the first cell to put them somewhere
else.

## Results

The scores saved in notebooks 1 and 2 come from one seed on part of the dataset. The scores in the
vector notebooks come from damaged targets and not from a model. They all show the workflow and
don't measure what the models can do. For measured comparisons see
[`experiments/`](../experiments/).
