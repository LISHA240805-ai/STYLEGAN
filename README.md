# StyleGAN on CIFAR-10

## Introduction

This project is a compact, educational implementation of a **StyleGAN-style Generative Adversarial Network (GAN)** built from scratch in TensorFlow/Keras. It generates 32×32 colour images and demonstrates the core ideas that make StyleGAN different from a traditional GAN.

**Key StyleGAN concepts implemented:**

- **Mapping Network:** An 8-layer fully connected network that transforms a latent vector `z` into an intermediate latent space `w` (dimension 128). The input is normalized (pixel norm) and uses LeakyReLU activations.
- **Learned Constant Input:** The generator starts from a learned 4×4×128 constant tensor instead of feeding noise directly into the first layer.
- **Adaptive Instance Normalization (AdaIN):** Style vectors modulate the feature maps at every layer, controlling the look of the generated image.
- **Per-layer Noise Injection:** Learnable noise-strength weights add stochastic detail (fine variation) to the features.
- **Progressive resolution blocks:** Synthesis goes 4×4 → 8×8 → 16×16 → 32×32 with 8 styled convolution layers (channels 128, 128, 64, 32).
- **Discriminator:** A convolutional network with 4 strided-conv blocks (32, 64, 128, 128 filters) and a dense output head.
- **Non-saturating logistic GAN loss** (softplus), trained with Adam (lr = 2e-4, β₁ = 0.0, β₂ = 0.99).

## Procedure used

1. Loads and preprocesses CIFAR-10 images (scaled to [-1, 1]).
2. Builds the Style Generator and Discriminator.
3. Trains the GAN for 10 epochs (batch size 64) with a custom `@tf.function` training step.
4. Plots the generator and discriminator loss curves.
5. Displays a grid of generated samples from a fixed latent seed.
6. Demonstrates **style mixing**, where early-layer styles (coarse structure) from image A are combined with later-layer styles (fine details) from image B at layers 2, 4, and 6.
7. Demonstrates **latent-space interpolation**, which smoothly morphs between two generated images across 9 steps.

## Dataset Used

**CIFAR-10** (loaded via `keras.datasets.cifar10`)

| Property | Details |
|---|---|
| Source | Canadian Institute For Advanced Research (Krizhevsky & Hinton) |
| Image size | 32 × 32 pixels, RGB (3 channels) |
| Total training images | 50,000 (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Subset used here | **First 20,000 training images** (deterministic subset for a faster run) |
| Labels | Not used (unsupervised generation) |
| Preprocessing | Pixel values scaled from [0, 255] to [-1, 1] using `(x - 127.5) / 127.5` |
| Pipeline | `tf.data` with shuffling (seed 42), batch size 64, `drop_remainder=True`, and prefetching |

The dataset downloads automatically the first time the notebook runs.

## Libraries Used

| Library | Purpose |
|---|---|
| **TensorFlow** | Core deep learning framework (`tf.data`, `tf.function`, `GradientTape`) |
| **Keras** (`tf.keras`) | Model building: `keras.Model`, `layers.Dense`, `Conv2D`, `LayerNormalization`, `LeakyReLU`, Adam optimizer, CIFAR-10 loader |
| **NumPy** | Array handling and random seeding |
| **Matplotlib** | Plotting real images, generated samples, loss curves, style mixing and interpolation |
| **os, time, random** (Python standard library) | Environment config, training timing, reproducibility |

## Requirements

```
tensorflow>=2.16
numpy
matplotlib
```

> `LeakyReLU(negative_slope=...)` requires Keras 3 (TensorFlow 2.16+).

## How to Run

```bash
pip install tensorflow numpy matplotlib
jupyter notebook StyleGAN.ipynb
```

Run all cells in order. A GPU is recommended but not required, since the model is small and the dataset is a 20k-image subset.

## Notebook Structure

| Cell | Description |
|---|---|
| 1 | Imports, random seed setup, device check |
| 2 | CIFAR-10 loading, preprocessing, `tf.data` pipeline, sample preview |
| 3 | Model definitions: `MappingNetwork`, `AdaIN`, `StyledConv`, `StyleGenerator`, `Discriminator` |
| 4 | Training loop (10 epochs) with loss tracking |
| 5 | Loss curves and generated image grid |
| 6 | Style mixing demonstration |
| 7 | Latent-space interpolation |

## Results

- Training loss curves for the generator and discriminator
- An 8×8 grid of generated CIFAR-10-like images
- Style-mixing outputs (mix at layers 2, 4, 6)
- A 9-step latent interpolation between two random latent vectors

