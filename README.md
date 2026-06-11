Minor Project Repository

This repository contains the complete source code for my minor project. It is structured around several Jupyter notebooks, each representing a major breakthrough in my research—from basic medical classification to engineering state-of-the-art Vision-Language Models and Vision Mamba architectures from scratch.

Below is the detailed breakdown of every single file in this repository.


(Sem_6)

 1. minor_minor_final.ipynb
(Polyp Detection Pipeline: SemiDAViL + CARZero)

 What I Have Done: I built a complete, two-stage clinical AI pipeline for colorectal polyp detection. Phase 1 performs pixel-perfect segmentation using a Vision-Language Model (SemiDAViL), and Phase 2 performs a zero-shot classification (CARZero) to provide a final diagnosis. I implemented custom PyTorch DataLoaders to mathematically balance the Kvasir-v2 Dataset.

 Architectural Flow:

Input: A raw colonoscopy image is uploaded.
Phase 1 (Segmentation): The image enters a frozen CLIP Vision Encoder while the text "a photo of a polyp" enters the Text Encoder.
Cross-Attention Fusion: The visual and text features fuse together using Dense Language Guidance. The model outputs a black-and-white Segmentation Mask.
Transition: The mask isolates the polyp and deletes the healthy background.
Phase 2 (Classification): The isolated polyp features are extracted and passed through my custom Bidirectional Node Mask.
Alignment & Output: The masked features are aligned against text embeddings ("Normal Colon" vs "Polyp Present") via Cross-Attention to calculate the final diagnosis.
Uniqueness: Standard AI models rely entirely on visuals. My architecture is unique because it is heavily guided by Language. The AI physically locates the polyp by comparing the pixels to the English definition of a polyp. Furthermore, I bypassed standard Linear Classifiers, instead using mathematical semantic distance (Cross-Attention) to classify the image.

 Why it is Better: Standard Vision Transformers use Global Attention, which gets confused by surgical fluids and camera glare on the opposite side of the image. By engineering the Bidirectional Node Mask, I mathematically forced the model to only look at its immediate neighboring pixels. This localizes the focus to the physical edges of the colon wall, making it vastly superior at handling medical noise.

Results:

Achieved an exceptional 98% classification accuracy on the Kvasir-v2 Dataset.
Successfully demonstrated Zero-Shot Cross-Domain capability (trained on CVC-ColonDB, successfully tested on ETIS-LaribPolypDB without retraining).

 2. m_2.ipynb
(Vision Mamba Implementation from Scratch)

 What I Have Done: Rather than relying on heavy, pre-compiled CUDA libraries, I mathematically constructed a minimal Vision Mamba (VMamba) architecture entirely from the ground up using pure PyTorch. I also integrated kagglehub to pull and process the massive ImageNet-Mini (1000 classes) dataset.

Architectural Flow:

Input: ImageNet-Mini images pass through PyTorch augmentations and DataLoaders.
PatchEmbed: The image is chopped into patches and linearly projected into embeddings.
VSSBlock (The Core Engine): The tokens enter the Vision State Space block, passing through a LayerNorm.
Selective Scanning (SS2D): The tokens enter my custom Simple Selective Scan, using 1×3 and 3×1 directional convolutions to scan the 2D image sequentially in multiple directions.
PatchMerge: The spatial resolution is halved and channel dimensions are doubled.
Output: Tokens are globally pooled and passed to a linear classifier to predict the ImageNet class.
 Uniqueness: Most SSM (State Space Model) implementations require complex C++ kernel integrations like mamba_ssm. My implementation is unique because I simulated the 2D Selective Scan mechanism using pure PyTorch 1D directional convolutions, making the architecture incredibly readable, educational, and easy to run on any GPU.

Why it is Better: Standard Vision Transformers (ViTs) suffer from an O(N 2) computational bottleneck because every pixel must calculate attention with every other pixel. My custom VMamba implementation is better because it processes global spatial features with a linear O(N) complexity, drastically reducing computational overhead while maintaining deep hierarchical understanding.

 Results:
Successfully proved that a pure-PyTorch 2D Selective Scan approach can effectively learn hierarchical visual representations on the challenging ImageNet-Mini dataset without requiring self-attention.



(Sem_5)

3. minor-project.ipynb & minor-project-2.ipynb
(Iterative Medical Segmentation Research)

What I Have Done: These notebooks document the foundational iterations of my medical segmentation research. Before finalizing SemiDAViL, I used these files to build robust PyTorch Dataset classes, handle heavy image augmentations, and experiment with loss functions on medical data.

 Architectural Flow:
 
Preprocessing: Raw endoscopy images are aggressively augmented (flips, rotations, color jittering) to artificially expand the dataset.
Student-Teacher Loop: Unlabeled data is passed to a Teacher model to generate pseudo-labels.
Loss Calculation: The Student model predictions are compared against ground-truth labels and Teacher pseudo-labels.
Backpropagation: The Student weights are updated via gradient descent, while Teacher weights are updated via Exponential Moving Average (EMA).

Instead of using standard Binary Cross Entropy (BCE) loss, I implemented DyCE Loss (Dynamic Cross-Entropy). Medical images have extreme class imbalance (the polyp is tiny, the background is huge). DyCE loss is better because it dynamically finds the hardest 20% of pixels (usually the blurry edges of the polyp) and forces the AI to focus its learning there.

Results:
Established a robust training pipeline that successfully leveraged unlabeled data, resulting in highly accurate bounding masks even when trained on extremely small labeled datasets.

4. training_on_chest_x_ray_Dataset.ipynb
(Baseline Pulmonary Anomaly Classification)

What I Have Done: This file represents the origin point of my medical imaging research. I built a foundational deep learning classifier to detect pulmonary anomalies and pneumonia from raw thoracic Chest X-Ray scans.

Architectural Flow:

Data Split: The dataset is strictly partitioned into 70% Training, 20% Validation, and 10% Testing to prevent data leakage.
Normalization: The grayscale X-Ray images are normalized to standardize pixel intensity across the dataset.
Feature Extraction: The images are passed through a Convolutional Neural Network (CNN) to extract hierarchical thoracic features.
Classification: A fully connected layer outputs the probability of disease presence.

This pipeline is heavily structured around preventing overfitting. By enforcing a strict 70/20/10 split and implementing robust validation checks, the model ensures that the reported accuracy represents true generalization rather than just memorizing the training data.

Results:

Successfully proved the viability of automated feature extraction on Chest X-Rays, serving as the foundational proving ground for my future, more complex Vision-Language architectures.
