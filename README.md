<div align="center">

# GaussianOcc3D: A Gaussian-Based Adaptive Multi-modal 3D Occupancy Prediction

**A. Enes Doruk** · **Hasan F. Ates**

[**Paper (arXiv)**](https://arxiv.org/abs/2601.22729) · [**Paper (HTML)**](https://arxiv.org/html/2601.22729v1)

</div>

## Overview

GaussianOcc3D is a camera–LiDAR fusion framework for **3D semantic occupancy prediction**. It represents a driving scene using learnable, anisotropic **3D Gaussian primitives** rather than relying exclusively on dense voxel features or compressed bird's-eye-view representations. Camera appearance and LiDAR geometry are aggregated in a shared Gaussian space, fused according to their cross-modal consistency, and refined to predict a dense semantic occupancy grid.

The framework introduces four main components:

- **LiDAR Depth Feature Aggregation (LDFA):** lifts sparse LiDAR features into Gaussian primitives using depth-wise deformable sampling and gated global context aggregation.
- **Entropy-Based Feature Smoothing (EBFS):** measures camera–LiDAR feature disagreement using bidirectional cross-entropy and reduces modality-specific noise before fusion.
- **Adaptive Camera–LiDAR Fusion (ACLF):** combines dual-stream cross-attention, learned fusion gates, and consistency-aware reweighting to adaptively balance geometric and semantic cues.
- **Gauss-Mamba Head:** uses selective state-space modeling on ordered Gaussian features to capture long-range spatial context efficiently before Gaussian refinement and voxel splatting.

## Framework

The overall pipeline comprises the following stages:

1. Extract multi-scale image features using a **ResNet-101-DCN + FPN** camera encoder and sparse geometric features using a **LiDAR sparse encoder**.
2. Initialize Gaussian primitives and aggregate camera features through multi-view sampling and LiDAR features through **LDFA**.
3. Apply **EBFS** to reduce cross-modal feature discrepancies, then fuse the features with **ACLF**.
4. Refine the fused Gaussian representation and model global dependencies with the **Gauss-Mamba Head**.
5. Convert the refined 3D Gaussian representation into a dense semantic occupancy prediction through **Gaussian-to-voxel splatting**.

See the [paper](https://arxiv.org/abs/2601.22729) for the full architecture and module details.

## Main Results

The following are the **mIoU scores reported in our paper** for the multi-modal GaussianOcc3D model:

| Benchmark | Evaluation split | mIoU (%) |
| :-- | :-- | --: |
| Occ3D (nuScenes) | Validation | **49.4** |
| SurroundOcc (nuScenes) | Validation | **28.9** |
| SemanticKITTI | Test | **25.2** |

On the SurroundOcc validation set, GaussianOcc3D achieves **27.1% mIoU in rainy scenes** and **15.9% mIoU at night**, demonstrating its robustness under challenging sensing conditions. The reported model uses **25,600 Gaussian primitives**; detailed per-class evaluations, ablations, and efficiency comparisons are provided in the paper.

## Model Configurations

The repository includes the following GaussianOcc3D configurations:

| Input modalities | Configuration |
| :-- | :-- |
| Camera + LiDAR | [`projects/configs/GaussianOcc3D/GaussianOcc3D-Multimodal_25600.py`](projects/configs/GaussianOcc3D/GaussianOcc3D-Multimodal_25600.py) |
| Camera-only | [`projects/configs/GaussianOcc3D/GaussianOcc3D-Camera_25600.py`](projects/configs/GaussianOcc3D/GaussianOcc3D-Camera_25600.py) |
| LiDAR-only | [`projects/configs/GaussianOcc3D/GaussianOcc3D-LiDAR_25600.py`](projects/configs/GaussianOcc3D/GaussianOcc3D-LiDAR_25600.py) |

**Note:** Before training, adapt the dataset paths, annotation paths, and pretrained backbone checkpoint path in the chosen configuration to your local environment.

# Getting Started
**1. Create a conda virtual environment and activate it.**
```shell
conda create -n OccMamba python=3.9 -y
conda activate OccMamba
```

**2. Install PyTorch (tested on torch==1.13.1 & cuda=11.7/11.8).**
```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

**3. Install gcc>=5 in conda env.**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**4. Install mmcv, mmdet, mmseg and mmdet3d. These versions is not mandatory, but code changes may be required**
```shell
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmdet==2.26.0 mmsegmentation==0.30.0 mmdet3d==1.0.0rc5
```

**5. Install other dependencies.**
```shell
pip install timm open3d-python PyMCubes spconv-cu117 fvcore IPython
pip install causal-conv1d==1.2.0.post2 mamba-ssm==1.2.0.post1 # other versions are not tested
pip install numpy==1.22.4 yapf==0.40.1 # downgrade
```

**6. Install occupancy pooling, same to the OpenOccupancy.**
```shell
export PYTHONPATH="."
python setup.py develop
```

**7. Fix some dependence crash.**
```shell
cp dependence/dag.py path_to_conda_env/OccMamba/lib/python3.9/site-packages/networkx/algorithms/dag.py
```

**8. Prepare the dataset by following the [instructions](https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md) in OpenOccupancy.**

# Training and Inference
**1. Training examples.**
```shell
bash run.sh $PATH_TO_CFG $GPU_NUM
bash run.sh ./projects/configs/GaussianOcc3D/GaussianOcc3D-Multimodal_25600.py 8
```

**2. Inference examples. If you want to save prediction results, use `--show` and `--show-dir`.**
```shell
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM --show --show-dir $PATH_TO_SAVE
```

# Visualization
**Visualization example.**
```
python tools/visual.py $PATH_TO_NPY
```

## Pretrained Weights

The supplied repository archive does not include GaussianOcc3D pretrained checkpoints. In particular, the pretrained-model links from the original OccMamba README do **not** correspond to GaussianOcc3D and have therefore been removed.

## Code Organization

The main GaussianOcc3D implementation can be found in:

- `projects/occ_plugin/occupancy/detectors/gaussianocc3d.py` — GaussianOcc3D model and multi-modal processing.
- `projects/occ_plugin/occupancy/backbones/gaussian_encoder/depth_aggregation.py` — LiDAR depth feature aggregation (LDFA).
- `projects/occ_plugin/occupancy/fuser/smoothnization.py` — entropy-based feature smoothing (EBFS).
- `projects/occ_plugin/occupancy/fuser/adaptive_fusion.py` — adaptive camera–LiDAR fusion (ACLF).
- `projects/occ_plugin/occupancy/backbones/gaussian_encoder/refine_module.py` — Gaussian refinement, including Mamba-based blocks.
- `projects/occ_plugin/occupancy/dense_heads/gaussian_head.py` — Gaussian occupancy head.

## Citation

If you find GaussianOcc3D useful for your research, please cite:

```bibtex
@misc{doruk2026gaussianocc3d,
  title         = {GaussianOcc3D: A Gaussian-Based Adaptive Multi-modal 3D Occupancy Prediction},
  author        = {Doruk, A. Enes and Ates, Hasan F.},
  year          = {2026},
  eprint        = {2601.22729},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2601.22729}
}
```

## Acknowledgements

This implementation builds upon and draws inspiration from the open-source ecosystems of [OccMamba](https://github.com/USTCLH/OccMamba), [GaussianFormer](https://github.com/huang-yh/GaussianFormer), and [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy). We thank the authors for making their research and code available.
