# minor(sem-5)
What we Have Done
 1. Downloaded & prepared ImageNet-Mini dataset
Used kagglehub to automatically download Imagenet-Mini (1000 classes).
Loaded dataset with PyTorch ImageFolder.
Applied standard ImageNet transformations.
Created train/val dataloaders with batching and shuffling.
 2. Built a minimal VMamba-like model from scratch
Created a simple but functional VMamba-style architecture, including:
PatchEmbed (Conv2d → patch tokenization)
SimpleSelectiveScan (1×3 and 3×1 conv-based directional scanning)
VSSBlock (LayerNorm → Selective Scan → MLP → Residual)
PatchMerge (downsampling & channel doubling)
A small hierarchical encoder:
64 → 128 → 256 channels
Global average pooling + Linear classifier
 3. Implemented training pipeline
Custom training loop using AdamW + CrossEntropyLoss
Limited training to 50 images per epoch (quick demo)
Printed train/val accuracy each epoch
Saved model weights:
vmamba_weights.pth
4. Implemented full inference pipeline
Upload any image in Colab
Transform → forward pass → prediction
Display top-3 classes with:
Human-readable ImageNet label,
WordNet ID (e.g., n02099601),
Probability score
