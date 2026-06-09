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
> Poznań University of Technology, Faculty of Computing and Telecommunications (2026)

</div>

---

## 🎯 What Does It Do?

The system takes a **heavily degraded image** of overlapping geometric structures and reconstructs it as a set of **clean, parametric ellipses** — each described by position, size, rotation, and colour. Unlike a standard autoencoder that outputs raw pixels, this model outputs *symbolic parameters* that a differentiable renderer then turns into an image. The entire pipeline is end-to-end trainable.

<p align="center">
  <img src="master%20thesis/figures/grid_reconstruction_results.png" width="700" alt="Input → Target → Network Prediction"/>
  <br/>
  <em>Left: noisy input &nbsp;|&nbsp; Centre: clean target &nbsp;|&nbsp; Right: network prediction</em>
</p>

The approach sits at the intersection of three fields:

<p align="center">
  <img src="master%20thesis/figures/venn_final_fixed_pos.png" width="420" alt="Venn diagram: Deep Learning × Computer Graphics × Symbolic AI"/>
</p>

---

## ✨ Key Features

- **Two decoder architectures** — pick the one that suits your data:
  - **Sequential GRU decoder** — generates ellipses one by one as a chain; solves the halting problem via a learned stop token.
  - **16×16 Spatial Grid decoder** — each grid cell independently predicts an ellipse; handles arbitrary bifurcations, X-crossings, and branching topologies that trip up sequential models.

- **Differentiable 2D Renderer** — a closed-form, analytical rendering engine based on sigmoid functions. Gradients flow directly from pixel space back to ellipse parameters, enabling end-to-end learning with no separate rendering step.

- **On-the-fly data generator** — the system is fully self-contained. It generates stochastic paths and bifurcations in RAM, then applies extreme visual degradations (Gaussian noise, blur, pixel dropout) that simulate real medical imagery such as ultrasound and angiography scans.

- **Orthogonal loss function** — prevents *Zero-Collapse* (the model predicting blank images) by separating topology/geometry learning (Dice coefficient) from photometric learning (MSE in RGB space).

- **100% Reproducibility** — a fixed random seed (42) is applied globally so that every training run produces identical loss curves and weight convergence.

---

## 🏗️ Architecture

```
Input Image (128×128×3)
        │
        ▼
 ┌─────────────┐
 │  CNN Encoder │  4× Conv2d → Flatten → Linear → Tanh
 │  (Shared)   │  Output: latent context vector C (128-dim)
 └──────┬──────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
 GRU       Grid (16×16)
 Decoder   Decoder
   │         │
   └────┬────┘
        │  Sequence of ellipse parameter vectors [x, y, rx, ry, θ, R, G, B, α]
        ▼
 ┌─────────────────────┐
 │  Differentiable 2D  │  Sigmoid-based analytical renderer
 │  Renderer           │  Pixel grid → composited RGB image
 └─────────────────────┘
        │
        ▼
 Reconstructed Image (128×128×3)
```

### Ellipse parameter vector (9 values per primitive)

| Index | Parameter | Range | Description |
|-------|-----------|-------|-------------|
| 0 | `cx` | [0, 1] | Centre X |
| 1 | `cy` | [0, 1] | Centre Y |
| 2 | `rx` | [0, 1] | Radius X (semi-axis) |
| 3 | `ry` | [0, 1] | Radius Y (semi-axis) |
| 4 | `θ`  | [0, 1] → [0, π] | Rotation angle |
| 5 | `R`  | [0, 1] | Red channel |
| 6 | `G`  | [0, 1] | Green channel |
| 7 | `B`  | [0, 1] | Blue channel |
| 8 | `α`  | [0, 1] | Existence probability (stop signal) |

---

## 📊 Datasets

Three procedurally generated datasets — no external data needed.

| Dataset | Model | Description |
|---------|-------|-------------|
| `geometric` | GRU | Sequential snake-like paths of tangent ellipses; 5 complexity levels |
| `bifurcation` | Grid | Intersecting line segments with X-crossings and T-junctions |
| `HardcoreUSG` | Both | Any base dataset wrapped with Gaussian noise + blur + dropout |

<p align="center">
  <img src="master%20thesis/figures/dataset_capabilities.png" width="700" alt="Five levels of dataset complexity for the GRU model"/>
  <br/>
  <em>GeometricDataset — five complexity levels (A→E) used during GRU training</em>
</p>

<p align="center">
  <img src="master%20thesis/figures/dataset_complex_topologies.png" width="600" alt="BifurcationDataset — line crossings"/>
  <br/>
  <em>BifurcationDataset — intersecting structures for the Grid decoder</em>
</p>

<p align="center">
  <img src="master%20thesis/figures/robustness_samples.png" width="700" alt="HardcoreUSGDataset — noise degradations"/>
  <br/>
  <em>HardcoreUSGDataset — three degradation types applied at severity 0.7</em>
</p>

---

## 📁 Repository Structure

```
Neurosymbolic-Autoencoders-for-Modelling-of-2D-Structures/
│
├── main.py                  # CLI entry point (argparse)
├── requirements.txt         # Python dependencies
│
├── core/
│   ├── renderer.py          # Differentiable 2D renderer (sigmoid-based)
│   └── losses.py            # Composite loss: Dice + MSE + BCE
│
├── engine/
│   └── trainer.py           # Training loops — standard + 3-phase (GRU)
│
├── models/
│   └── networks.py          # CNN encoder, GRU decoder, Grid decoder
│
├── utils/
│   ├── dataset.py           # Data generators (Geometric, Bifurcation, HardcoreUSG)
│   └── visualization.py     # Reconstruction previews & t-SNE plots
│
└── master thesis/
    ├── main.pdf             # Full thesis document
    └── figures/             # All result figures used in thesis and README
```

---

## ⚙️ Requirements

- Python 3.10+
- CUDA-capable GPU recommended (CPU works but is slow for training)

Install all dependencies:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

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

### Clone

```bash
git clone https://github.com/pchumski/Neurosymbolic-Autoencoders-for-Modelling-of-2D-Structures.git
cd Neurosymbolic-Autoencoders-for-Modelling-of-2D-Structures
pip install -r requirements.txt
```

### Train — Grid decoder (recommended starting point)

```bash
python main.py --model-type grid --dataset-type bifurcation --epochs 80
```

### Train — GRU decoder on simple paths

```bash
python main.py --model-type gru --dataset-type geometric --epochs 60
```

### Train with noise (simulate ultrasound data)

```bash
python main.py --model-type grid --dataset-type bifurcation --noise-severity 0.7 --epochs 100
```

---

## 📋 Full CLI Reference

| Argument | Choices / Default | Description |
|---|---|---|
| `--model-type` | `gru` \| `grid` | **Required.** Decoder architecture |
| `--dataset-type` | `geometric` \| `bifurcation` · `geometric` | Training data generator |
| `--noise-severity` | `0.0–1.0` · `0.0` | Visual degradation intensity (0 = off) |
| `--epochs` | int · `60` | Total training epochs |
| `--batch-size` | int · `32` | Batch size |
| `--lr` | float · `2e-4` | Learning rate |
| `--device` | `cpu` \| `cuda` · `cuda` | Hardware target |
| `--num-samples` | int · `10000` | Number of generated training samples |
| `--grid-size` | int · `16` | Spatial grid resolution (Grid decoder only) |

After training, two files are saved automatically:

- `neurosymbolic_{model_type}_model.pth` — trained weights
- `test_results_{model_type}.png` — side-by-side reconstruction preview

---

## 📈 Results

### Training curve

<p align="center">
  <img src="master%20thesis/figures/loss_history.png" width="500" alt="Training loss — converges within ~10 epochs"/>
  <br/>
  <em>Loss converges within the first 10 epochs and stabilises cleanly</em>
</p>

### Grid decoder — reconstruction quality

<p align="center">
  <img src="master%20thesis/figures/grid_reconstruction_results2.png" width="700" alt="More reconstruction examples including X-crossings"/>
  <br/>
  <em>The Grid decoder successfully handles X-crossings (row 1) and single-segment cases (rows 2, 4)</em>
</p>

### Latent space — t-SNE (GRU model)

<p align="center">
  <img src="master%20thesis/figures/tsne_latent_space_gru.png" width="580" alt="t-SNE projection coloured by object count"/>
  <br/>
  <em>t-SNE of the 128-dim latent space coloured by number of objects — the encoder organises complexity into a smooth gradient</em>
</p>

---

## 📄 Links

| Resource | Link |
|---|---|
| 📖 Full Master's Thesis (PDF) | [master thesis/main.pdf](master%20thesis/main.pdf) |
| 🧩 PyTorch | [pytorch.org](https://pytorch.org/) |

---

<div align="center">

*Master's Thesis — Poznań University of Technology*
**Author: Paweł Chumski · 2026**

</div>
