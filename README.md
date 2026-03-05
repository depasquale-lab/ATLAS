# ATLAS
### **A**nimal **T**rajectory **L**atent **A**lignment **S**pace

> A community framework for mapping behavioral time series — keypoints, pose trajectories, or movement statistics — into a shared latent space across animals, subjects, and experimental conditions.

---

## Overview

Understanding behavior across individuals, species, and recording setups requires a common language. ATLAS provides that language by learning a shared latent representation of multidimensional behavioral time series using a **Variational Autoencoder (VAE)** framework.

Whether your data comes from markerless pose estimation, motion capture, or hand-crafted behavioral statistics, ATLAS encodes it into a unified latent space — enabling cross-animal comparison, alignment, and analysis that was previously impractical.

```
raw time series (keypoints / behavioral statistics)
            │
            ▼
    ┌───────────────┐
    │  ATLAS  (VAE) │   ←  shared latent space
    └───────────────┘
            │
            ▼
   aligned representation
   across animals & conditions
```

---

## Motivation

Behavioral neuroscience faces a fundamental challenge: no two animals move in exactly the same way. Differences in body size, recording angle, and individual variation make it hard to directly compare neural or behavioral data across subjects. ATLAS is built to bridge this gap — learning the structure of behavior in a representation that generalizes.

This project is in active development and aims to grow into a community resource for anyone working at the intersection of **behavior, neuroscience, and machine learning**.

---

## Key Features

- **Flexible inputs** — accepts any multidimensional behavioral time series: pose keypoints, skeletal trajectories, ethogram features, kinematic statistics
- **VAE backbone** — probabilistic latent space with smooth, structured representations suitable for downstream analysis
- **Cross-animal alignment** — designed from the ground up for mapping behavior across individuals and species
- **Modular architecture** — swap in your own encoder, decoder, or loss components
- **Built on PyTorch** — familiar, extensible, and research-ready

---

## Installation

```bash
git clone https://github.com/yourusername/ATLAS.git
cd ATLAS
pip install -e .
```

**Requirements:**
- Python >= 3.9
- PyTorch >= 2.0

---

## Quickstart

```python
from atlas import ATLASVAE

# Load your behavioral time series (N_samples x T_timepoints x D_features)
import torch
data = torch.load("my_keypoints.pt")

# Initialize and train the model
model = ATLASVAE(input_dim=34, latent_dim=16, seq_len=100)
model.fit(data)

# Encode into the shared latent space
z = model.encode(data)
```

---

## Data Format

ATLAS expects behavioral time series of shape:

```
(N, T, D)
  │  │  └─ feature dimensions (keypoints × axes, or behavioral statistics)
  │  └──── time steps
  └─────── samples / trials
```

Both **keypoint data** (e.g., from DeepLabCut, SLEAP, Anipose) and **behavioral statistics** (e.g., speed, limb angles, ethogram vectors) are supported.

---

## Roadmap

- [x] VAE core architecture
- [ ] Cross-animal alignment training objective
- [ ] Pretrained model weights for common behavioral assays
- [ ] Support for variable-length sequences
- [ ] Integration with DeepLabCut / SLEAP output formats
- [ ] Visualization and analysis utilities
- [ ] Benchmarks and example datasets

---

## Contributing

ATLAS is a community project. Contributions of all kinds are welcome — new architectures, datasets, tutorials, bug reports, or ideas.

1. Fork the repo and create a feature branch
2. Make your changes with tests where applicable
3. Open a pull request with a clear description

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Citation

If you use ATLAS in your research, please cite this repository while a formal publication is in preparation:

```bibtex
@software{atlas2025,
  title  = {ATLAS: Aligned Time-series Latent Access Space},
  author = {DePasquale, Brian and contributors},
  year   = {2025},
  url    = {https://github.com/yourusername/ATLAS}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
