# GCSCAN

**Graph-based Cellular Spatial Clustering Against Noise (GCSCAN)** identifies spatially clustered cellular structures, such as tertiary lymphoid structures (TLSs) and B cell follicles, in spatial single-cell data.

This repository provides the Python implementation (`gcscan.py`). A small example dataset and Jupyter notebook (`example_gcscan.ipynb`) are also provided as tutortials. For a detailed explanation of the method, see the accompanying manuscript.



## Setup

Use a Python environment with the following packages:

```bash
python -m pip install numpy pandas scipy scikit-learn networkx matplotlib scanpy seaborn pyarrow 
```

Note that scanpy, seaborn and pyarrow are used in file I/O or plotting, not required by GCSCAN itself.

These packages are also available via Conda.

To use, simply copy the gcscan.py file to a place that's accessible by your Python environment.

For the full workflow, including plots, see [`example_gcscan.ipynb`](example_gcscan.ipynb).

## Quick Start

### Core function

GCSCAN expects **one row per cell**, in matching order across inputs:

| Input | Format | Meaning |
| --- | --- | --- |
| `xy` | `(n_cells, 2)` array | Cell-center coordinates, here `cdf[['x', 'y']].values`. |
| `coords` | Length-`n_cells` collection of coordinate arrays | Cell boundary polygons, needed for boundary-based adjacency and polygon visualization. |
| `involved` | Length-`n_cells` array | Indicator of whether each cell belongs to a population associated with the target structure. |
| `iscore` | Length-`n_cells` Boolean array | Cells of interest within a "core" area inside detected structures, here GC B cells. |


Common settings

| Parameter | What to change it for |
| --- | --- |
| `edge_method` | `'delaunay'` for Delaunay-based adjacency, `'boundary'` for adjacency based on cell boundary separation, or `'center'` for radius-based center adjacency. |
| `radius` | Maximum center-to-center distance considered when constructing center/Delaunay edges. **See implementation note below.** |
| `boundary_dist_threshold` | Maximum separation between cell boundaries for retained edges in boundary mode. |
| `minimal_neighbor_count` | Minimum denominator used in calculating local involvement scores. |
| `watershed_initial_threshold` | Involvement threshold for selecting watershed seeds. |
| `watershed_min_weight` | Minimum edge weight for watershed expansion. |
| `ncut_threshold` | Normalized-cut threshold for attempting to split loosely connected structures. |
| `iscore` | Indicator for GC-related cells in downstream GC-area estimation. |



Output:

| Result | Contents |
| --- | --- |
| `E_df` | Cell adjacency edges and edge weights. You shouldn't need to inspect this in most cases |
| `V_df` | Per-cell involvement scores and clustering results. The `cluster_final` column contains final cluster IDs; `0` denotes background/unassigned cells. |
| `cut_df` | Candidate normalized-cut splits and associated statistics. |

### Summarize structures and assign TLS IDs

```python
GC_df = gcscan.calculate_GC_df(V_df, cdf)

st_data.obs['TLS_id'] = gcscan.assign_TLS_id(GC_df, cdf)
st_data.obs['TLS_id'] = st_data.obs['TLS_id'].astype('category')
```

`GC_df` includes the number of cells in each detected structure (`n_cell`), estimated area (`Area`), estimated GC area (`GC_area`), GC area fraction (`GC_ratio`), and boundary coordinates (`coords`). The area calculations use the selected GC cells and the helper functions in `gcscan.py`.

`assign_TLS_id()` assigns cells to detected structure polygons. Cells outside all polygons receive no TLS ID. 

The notebook also shows how to draw the structure contours over cell boundaries and color cells by `TLS_id`.




## Citation

If you use GCSCAN, please cite the accompanying manuscript.



