# Morphology-Aware ECG Reconstruction Framework for Missing Precordial Leads in 12-Lead ECGs

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.6.0-%23EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

> **Official implementation of "Morphology-Aware ECG Reconstruction Framework for Missing Precordial Leads in 12-Lead ECGs".**

---

## Table of Contents
- [Overview](#overview)
- [Environments](#environments)
  - [Requirements](#requirements)
- [Installation](#installation)
  - [Using Docker](#using-docker)
  - [Using Conda](#using-conda)
- [How to Run](#how-to-run)
- [Experiments](#experiments)
  - [Hyperparameters](#hyperparameters)
  - [Results](#results)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview
This repository provides the official code for the Morphology-Aware ECG Reconstruction Framework.

---

## Environments
- **Docker**: 27.0.1
- **CPU**: 128-core Intel(R) Xeon(R) Gold 6430
- **GPU**: NVIDIA GeForce RTX 4090 (24GB)
- **NVIDIA Driver**: 560.35.03

### Requirements
- Python 3.11
- PyTorch 2.6.0

---

## Installation

### Using Docker
We recommend starting from the official PyTorch Docker image:

```bash
docker pull pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
```

Then, install extra libraries as specified in `requirements.txt`.

### Using Conda

```bash
conda create -n morphrec python=3.11
conda activate morphrec
conda install pytorch==2.6.0 cudatoolkit=12.6 -c pytorch
git clone <repo-url>
cd MorphRec
pip install -r requirements.txt
```

---

## How to Run

```bash
python main.py configs/exp.yaml
```

---

## Experiments

### Hyperparameters

The following hyperparameters are explored in the configuration.
You can modify their values in the configuration file at `configs/exp.yaml` to experiment with different settings:

| Name         | Values                        |
|--------------|-------------------------------|
| `LAMBDA_GUID`  | `{1, 0.8, 0.4, 0.2, 0.1}`     |
| `LAMBDA_PATCH` | `{1, 0.8, 0.4, 0.2, 0.1}`     |
| `PATCH_SIZE`   | `{25, 50, 75}`                |
| `LR`           | `{1e-4, 2e-4, 4e-4}`          |

---

### Results

**Coming soon.**

---
