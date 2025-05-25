# fusion-blocks

A PyTorch-based implementation of modular fusion blocks designed for multimodal image fusion tasks.  
This repository includes implementations of key components from **DDCINet** and partial modules from **CMAFF**, such as CSM and DEM.

## 📌 Overview

This repository focuses on implementing and testing advanced fusion modules for tasks involving:
- Cloud removal
- SAR-optical fusion
- RGB-Thermal image fusion
- Multimodal image restoration

## 🧠 Implemented Modules

### ✅ DDCINet
- **CSCA**: Channel-Gated Spatial Cross Attention
- **CMDF**: Cross-Modality Difference Filter
- **DFF**: Dual Feature Fusion

### ✅ CMAFF (Partial)
- **CSM**: Cross-Scale Modality attention
- **DEM**: Differential Enhancement Module

> ⚠️ Note: Full CMAFF is not implemented; only CSM and DEM modules are currently available.
