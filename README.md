Kenya Contrastive Learning

Objective:

Cross-modal representation learning between Sentinel-1 and Sentinel-2 imagery.

Current Architecture:

- ConvNeXt dual encoder

- 14-channel Sentinel-2 input

- 3-channel Sentinel-1 input

- Projection heads

- InfoNCE loss

Dataset:

- 637 matched S1-S2 pairs

- 256x256 patches

Training:

python train.py

Evaluation:

python evaluate_embeddings.py
