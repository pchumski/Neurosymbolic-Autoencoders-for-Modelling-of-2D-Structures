<div align="center">

# 🧠 Neurosymbolic Autoencoder for 2D Structures

**PyTorch · CNN · GRU · Differentiable Renderer · Analysis-by-Synthesis**

*A deep learning system that decomposes noisy 2D visual structures into interpretable geometric primitives (ellipses)*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Completed](https://img.shields.io/badge/Status-Completed-success.svg)]()

> Official code repository for the Master's Thesis:
> *"Neurosymbolic Autoencoders for Modelling of Sequential 2D Structures"*
> Poznań University of Technology, Faculty of Computing and Telecommunications · 2026

</div>

---

## 🧭 Overview

The system takes a **heavily degraded image** (Gaussian noise, blur, pixel dropout) of 2D structures and reconstructs it as a set of clean, parametric **ellipses** — each described by position, size, rotation, and colour. Unlike a standard autoencoder that outputs raw pixels, this model outputs *symbolic parameters* that a differentiable renderer turns back into an image. The entire pipeline is end-to-end trainable via backpropagation.

This work sits at the intersection of three research fields:

<p align="center">
  <img src="master%20thesis/figures/venn_final_fixed_pos.png" width="420" alt="This work combines Deep Learning, Computer Graphics, and Symbolic AI"/>
</p>

---

## ✨ Two Decoder Variants

The system was developed in two phases, producing two complementary architectures:

### 🐍 Sequential GRU Decoder — for chain-like structures

The GRU decoder generates ellipses **one step at a time**, like drawing a snake. It uses a two-layer GRU with a teacher-forcing training schedule and a learned stop token to handle variable-length sequences.

<p align="center">
  <img src="master%20thesis/figures/reconstruction_comparison.png" width="680" alt="GRU model: target (left), reconstruction (centre), predicted ellipse contours (right)"/>
  <br/>
  <em>GRU decoder — target (left) · reconstructed image (centre) · predicted ellipse centres and contours (right) · object count correctly recovered in all cases</em>
</p>

### 🔲 16×16 Grid Decoder — for complex topologies

The Grid decoder replaces the time axis with a **spatial 16×16 grid**: each cell independently predicts one ellipse, enabling fully parallel, deterministic inference. This architecture was designed specifically to handle X-crossings and T/Y-bifurcations — topologies that cause recurrent models to loop or stop early.

<p align="center">
  <img src="master%20thesis/figures/grid_reconstruction_results.png" width="680" alt="Grid decoder: noisy input, clean target, network prediction"/>
  <br/>
  <em>Grid decoder on the bifurcation dataset — noisy input (left) · clean target mask (centre) · network prediction (right)</em>
</p>

<p align="center">
  <img src="master%20thesis/figures/grid_reconstruction_results2.png" width="680" alt="Grid decoder: more examples including X-crossings"/>
  <br/>
  <em>More examples — including challenging X-crossings (row 1) and multi-segment scenes</em>
</p>

---

## 🏗️ Architecture

Both decoders share the same CNN encoder but differ in their reasoning module and loss function.

```
Input Image (128×128×3)
        │
        ▼
 ┌─────────────────────────────────────┐
 │  CNN Encoder (shared)               │
 │  4× Conv2d(stride=2) → MLP → Tanh  │
 └──────┬──────────────────────────────┘
        │
   ┌────┴──────────────────────┐
   │                           │
   ▼                           ▼
Flat vector C (128-dim)    Spatial tensor (S×S×D)
        │                           │
   GRU Decoder                 Grid Decoder
   (sequential)                (parallel, 16×16)
        │                           │
        └─────────────┬─────────────┘
                      │
              Ellipse parameters
         [x, y, rx, ry, θ, R, G, B, α]
                      │
                      ▼
         ┌────────────────────────┐
         │  Differentiable 2D     │
         │  Renderer (sigmoid)    │
         └────────────┬───────────┘
                      │
             Reconstructed image
              (128×128×3)
```

### Ellipse parameter vector — 9 values per primitive

| Index | Parameter | Range | Meaning |
|-------|-----------|-------|---------|
| 0, 1 | `cx`, `cy` | [0, 1] | Ellipse centre (normalised) |
| 2, 3 | `rx`, `ry` | [0, 1] | Semi-axes (scaled by learnable `Srx`, `Sry` in Grid) |
| 4 | `θ` | [0, π] | Rotation angle |
| 5–7 | `R`, `G`, `B` | [0, 1] | Colour |
| 8 | `α` | [0, 1] | Existence probability / stop signal |

### Differentiable renderer

Classical rasterisation is discrete and blocks gradient flow. The renderer approximates the hard ellipse boundary with a sigmoid function, enabling end-to-end optimisation:

```
α(x,y) = σ((1 − d(x,y)) · s)    where d(x,y) = (x_rot/rx)² + (y_rot/ry)²
```

Pixels accumulate colour via alpha-blending over all primitives in the sequence.

---

## 📊 Datasets

All datasets are generated procedurally at runtime — no external data required.

| Class | Used with | Description |
|-------|-----------|-------------|
| `GeometricDataset` | GRU | Snake-like chains of tangent ellipses; 5 curriculum complexity levels |
| `BifurcationDataset12` | Grid | Randomly intersecting line segments with X and T/Y junctions |
| `HardcoreUSGDataset` | Both | Wraps any base dataset with Gaussian noise + blur + pixel dropout |

<p align="center">
  <img src="master%20thesis/figures/dataset_capabilities.png" width="700" alt="Five complexity levels of the GeometricDataset (GRU training)"/>
  <br/>
  <em>GeometricDataset — five curriculum levels (A→E) used for GRU training, ranging from simple baseline to chaotic multi-overlap</em>
</p>

<p align="center">
  <img src="master%20thesis/figures/dataset_complex_topologies.png" width="600" alt="BifurcationDataset — clean reference masks"/>
  <br/>
  <em>BifurcationDataset — clean reference masks (ground truth). The model receives a noise-degraded version of these as input</em>
</p>

<p align="center">
  <img src="master%20thesis/figures/robustness_samples.png" width="700" alt="HardcoreUSGDataset — three degradation types at severity 0.7"/>
  <br/>
  <em>HardcoreUSGDataset — original (row 1), Gaussian noise (row 2), blur (row 3), pixel dropout (row 4) — all at severity 0.7</em>
</p>

---

## 📁 Repository Structure

```
Neurosymbolic-Autoencoders-for-Modelling-of-2D-Structures/
│
├── main.py                   # CLI entry point (argparse)
├── requirements.txt          # Python dependencies
│
├── core/
│   ├── renderer.py           # Differentiable 2D renderer (sigmoid-based)
│   └── losses.py             # Loss functions: Dice (topology) + MSE (colour) + BCE (stop)
│
├── engine/
│   └── trainer.py            # Training loops — standard (Grid) + 3-phase teacher forcing (GRU)
│
├── models/
│   └── networks.py           # CNN encoder, GRU decoder, Grid decoder
│
├── utils/
│   ├── dataset.py            # Data generators (Geometric, Bifurcation, HardcoreUSG)
│   └── visualization.py      # Reconstruction previews & t-SNE analysis
│
└── master thesis/
    ├── main.pdf              # Full thesis (58 pages)
    └── figures/              # All result figures
```

---

## ⚙️ Requirements

- Python 3.10+
- CUDA-capable GPU recommended (CPU works but is slow)

```bash
pip install -r requirements.txt
```

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/pchumski/Neurosymbolic-Autoencoders-for-Modelling-of-2D-Structures.git
cd Neurosymbolic-Autoencoders-for-Modelling-of-2D-Structures
pip install -r requirements.txt
```

**Train the Grid decoder** (recommended starting point — no teacher forcing needed):
```bash
python main.py --model-type grid --dataset-type bifurcation --epochs 80
```

**Train the GRU decoder** on sequential snake data:
```bash
python main.py --model-type gru --dataset-type geometric --epochs 60
```

**Add noise** to simulate degraded sensor data:
```bash
python main.py --model-type grid --dataset-type bifurcation --noise-severity 0.7 --epochs 100
```

After training, two files are saved automatically:
- `neurosymbolic_{model_type}_model.pth` — trained weights
- `test_results_{model_type}.png` — reconstruction preview

---

## 📋 Full CLI Reference

| Argument | Choices / Default | Description |
|---|---|---|
| `--model-type` | `gru` \| `grid` · **required** | Decoder architecture |
| `--dataset-type` | `geometric` \| `bifurcation` · `geometric` | Data generator |
| `--noise-severity` | `0.0–1.0` · `0.0` | Degradation intensity (0 = clean) |
| `--epochs` | int · `60` | Training epochs |
| `--batch-size` | int · `32` | Batch size |
| `--lr` | float · `2e-4` | Learning rate (Adam) |
| `--device` | `cpu` \| `cuda` · `cuda` | Hardware target |
| `--num-samples` | int · `10000` | Generated training samples |
| `--grid-size` | int · `16` | Grid resolution (Grid decoder only) |

---

## 📈 Results

### Training stability — gradient clipping

<p align="center">
  <img src="master%20thesis/figures/loss_clipping.png" width="560" alt="Effect of gradient clipping on training stability"/>
  <br/>
  <em>Gradient clipping (blue) vs without (red dashed) — without clipping, the GRU training explodes repeatedly after epoch 20</em>
</p>

### Training convergence (Grid decoder)

<p align="center">
  <img src="master%20thesis/figures/loss_history.png" width="500" alt="Grid model training loss — converges within ~10 epochs"/>
  <br/>
  <em>Grid model loss curve — sharp convergence in the first 10 epochs, stable plateau thereafter</em>
</p>

### Latent space organisation — t-SNE (GRU model)

<p align="center">
  <img src="master%20thesis/figures/tsne_latent_space_gru.png" width="580" alt="t-SNE of the 128-dim GRU latent space, coloured by object count"/>
  <br/>
  <em>t-SNE projection of the GRU latent space coloured by number of objects — the encoder organises scene complexity into a smooth, continuous gradient without explicit supervision</em>
</p>

---

## ⚠️ Limitations

The system was trained entirely on **synthetic data**. When applied zero-shot to real medical images (retinal angiography), the model fails due to domain shift — outputting structurally incorrect reconstructions:

<p align="center">
  <img src="master%20thesis/figures/drive_zero_shot_failure.png" width="680" alt="Zero-shot failure on real angiography images"/>
  <br/>
  <em>Zero-shot transfer to real retinal angiograms — the model cannot bridge the gap from synthetic to real-world data without fine-tuning</em>
</p>

Bridging this domain gap through transfer learning or domain adaptation is the primary direction for future work.

---

## 📄 Links

| Resource | Link |
|---|---|
| 📖 Full Master's Thesis (PDF, 58 pages) | [master thesis/main.pdf](master%20thesis/main.pdf) |
| 🧩 PyTorch | [pytorch.org](https://pytorch.org/) |

---

<div align="center">

*Master's Thesis — Poznań University of Technology*
**Author: Paweł Chumski · 2026**

</div>
