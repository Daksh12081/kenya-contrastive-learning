# Kenya Forest Classification using CROMA

## Overview

This repository contains the preprocessing and downstream training pipeline for land cover classification using the pretrained CROMA foundation model on the SEN12MS dataset.

## Dataset

- Dataset: SEN12MS Spring
- Total Samples: 38,755
- Modalities:
  - Sentinel-1 SAR
  - Sentinel-2 Optical

## Classes

| ID | Class |
|----|----------------|
|0|Forest|
|1|Shrub|
|2|Crop|
|3|Grass|
|4|NonVegetation|

## Pipeline

SEN12MS TIFFs

↓

Dataset Loader

↓

Training Tensors

↓

CROMA Encoder

↓

Embeddings

↓

Multiclass Classifier

## Repository Structure

```
datasets/
scripts/
embeddings/
downstream/
configs/
```

## Requirements

```
pip install -r requirements.txt
```

## Current Status

- Dataset preprocessing ✅
- Tensor generation ✅
- GCS upload ✅
- Downstream classifier ✅
- Embedding generation ⏳





