<div align="center">

# AI4Habitats

**Enabling and assessing Sentinel-2 based mapping of Natura 2000 habitats<br>with an analysis-ready national-scale data set**

[![Paper](https://img.shields.io/badge/ESSD-in%20preparation-lightgrey?logo=readthedocs&logoColor=white)](https://doi.org/PAPER_DOI)
[![Data](https://img.shields.io/badge/data-PhenoRoam-2e7d32?logo=databricks&logoColor=white)](https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/6d03d161-4251-4fdf-abaf-85c557e93230)
[![Data licence](https://img.shields.io/badge/data%20licence-CC%20BY%204.0-blue)](https://creativecommons.org/licenses/by/4.0/)
[![Code licence](https://img.shields.io/badge/code%20licence-MIT-blue)](LICENSE)
<br>
[![Python](https://img.shields.io/badge/python-3.12-3776ab?logo=python&logoColor=white)](setup/Readme.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-ff6f00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![RAPIDS cuML](https://img.shields.io/badge/RAPIDS-cuML%2025.08-7400b8?logo=nvidia&logoColor=white)](https://docs.rapids.ai/api/cuml/stable/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Data](#-data) · [Models](#-models) · [Installation](#-installation) · [Quick start](#-quick-start) · [Citation](#-citation)

</div>

---

AI4Habitats is an analysis-ready benchmark for mapping **Natura 2000 habitat types (Annex I of the Habitats Directive) from Sentinel-2 time series**. This repository contains the code to load the data set, generate train/validation/test splits, and train and evaluate three families of models on it: 

## 📦 Data

The data set is published on [PhenoRoam](https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/6d03d161-4251-4fdf-abaf-85c557e93230) (DOI: `DATA_DOI`) under CC BY 4.0; the attribution notices of the original data providers in `licence.md` apply.

|Feature|Description|
|---|---|
| Coverage | Germany, ≈ 32,800 patches of 256 × 256 px at 10 m (2.56 km × 2.56 km) |
| Inputs | Sentinel-2 L3A monthly composites (DLR), 10 bands (B2–B8A, B11, B12) × 24 months (March–October of three consecutive years) |
| Labels | harmonised habitat information of the federal states, encoded with the full Annex I legend (233 habitat types) plus background |

Download it with the downloader script (Python ≥ 3.9, no extra packages; downloads resume and are checksum-verified):

```bash
wget https://phenoroam.phenorob.de/file-uploader/download/public/30205-download_ai4habitats.py -O download_ai4habitats.py
python download_ai4habitats.py --toy                       # small toy subset to get started
python download_ai4habitats.py --out /path/to/big/disk     # full data set (≈ 1 TB)
python download_ai4habitats.py --help
```

The downloaded folder is self-contained; all paths in `patches_meta.csv` are relative to it:

```
<data_dir>/
├── patches_meta.csv              # one row per patch: patch_id, lat, lon, central_date, path_input, path_label
├── s2_normalization_stats.json   # channel-wise mean / std used for input standardisation
├── lrt_legend.csv                # habitat codes, names and colours
├── lrt_harmonized.gpkg           # harmonised habitat polygons (vector)
├── sentinel/patch_<id>.nc        # reflectances (band, time, y, x)
└── labels/patch_<id>.nc          # lrt_code, aoi_mask and covariates (y, x)
```

## 🧪 Models

| Model family | Input | Libraries |
|---|---|---|
| **Patchwise** deep semantic segmentation (U-Net, DeepLabV3+, SegFormer, DPT, …) | 256 × 256 px patches, all bands × time steps as channels | PyTorch, segmentation-models-pytorch |
| **Pixelwise** deep time-series classification (FCN, InceptionTime, MACNN, TapNet, MVTS, ConvTran, …) | per-pixel band × time stacks | TensorFlow / PyTorch via sktime |
| **Pixelwise** machine learning (random forest, kNN, SVM, logistic regression) | flattened per-pixel band × time features | RAPIDS cuML |

All families share the same data loading, label hierarchy, splits and evaluation metrics, so their results are directly comparable.

## 🔧 Installation

All three model families run in one conda environment with CUDA-compatible builds of PyTorch, TensorFlow and cuML. Requirements: Linux x86_64, an NVIDIA GPU (driver for CUDA 12.4), conda and git.

```bash
git clone GITHUB_URL ai4habitats && cd ai4habitats
bash setup/env_setup.sh
conda activate ai4habitats
python setup/env_verify.py
```

See [setup/Readme.md](setup/Readme.md) for further options.

> Training requires an NVIDIA GPU. Data loading, splitting, evaluation and plotting also work on CPU.

## 🚀 Quick start

See [`model_demo.ipynb`](model_demo.ipynb) for a walk-through on the whole library.  
Below just a brief demo:

```python
# load repository
import sys; sys.path.append("path/to/ai4habitats")    
from models.patchwise.train import Trainer

# init trainer
trainer = Trainer(
    meta_file="path/to/data/patches_meta.csv",
    sel_task="lvl2_bec",
    model_name="Unet",
    encoder_name="resnet18",
    loss="CrossEntropyLoss",
    max_epochs=30,
    save_dir="results/unet_lvl2_bec",
    wandb_project=None,
)
trainer.prep_model()

# train & evaluate the model
trainer.train_model()
```

## 📖 Citation

CITATION_PLACEHOLDER
If you use the data set or the code, please cite:

```bibtex
@article{ai4habitats,
  title   = {AI4Habitats: Enabling and assessing Sentinel-2 based mapping of Natura~2000 habitats
             with an analysis-ready national-scale data set},
  author  = {AUTHORS},
  journal = {Earth System Science Data},
  year    = {YEAR},
  note    = {in preparation},
  doi     = {PAPER_DOI}
}
```

## ⚖️ Licence

- **Code:** MIT License, see [LICENSE](LICENSE).
- **Data:** CC BY 4.0, see data repositories `licence.md`.

## 🙏 Acknowledgements

ACKNOWLEDGEMENTS_PLACEHOLDER