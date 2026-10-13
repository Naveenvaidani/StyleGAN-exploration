# StyleGAN Exploration on CIFAR-10

![Framework](https://img.shields.io/badge/framework-TensorFlow%2FKeras-orange)
![Dataset](https://img.shields.io/badge/dataset-CIFAR--10-blue)
![Task](https://img.shields.io/badge/task-image%20generation-2f855a)
![Notebook](https://img.shields.io/badge/interface-Jupyter%20Notebook-f37626)

An educational, research-oriented implementation of a **StyleGAN-inspired generative adversarial network (GAN)** trained on CIFAR-10. The experiment investigates how an intermediate latent representation and layer-wise style modulation affect image synthesis, and demonstrates two common latent-space operations: **style mixing** and **latent interpolation**.

The complete implementation is contained in [`StyleGAN.ipynb`](StyleGAN.ipynb). It is designed to make the main ideas behind style-based generation inspectable in a single notebook: dataset preparation, model construction, adversarial optimization, visualization, and latent-space analysis.

> **Scope.** This is a compact pedagogical implementation inspired by StyleGAN. It is not a reproduction of the complete StyleGAN, StyleGAN2, or StyleGAN3 training systems: it does not implement equalized learning rate, path-length regularization, progressive growing, the StyleGAN2 weight-demodulation formulation, or a large-scale training pipeline.

## Abstract

Generative adversarial networks learn to synthesize data by training two neural networks in competition. A generator maps a random latent vector to an image, while a discriminator estimates whether an image is real or generated. This project explores a style-based alternative to feeding the latent vector directly into the generator. A mapping network transforms the input latent vector $z$ into an intermediate vector $w$, and each styled convolution uses $w$ to control feature statistics through adaptive instance normalization (AdaIN).

The experiment uses a learned $4 \times 4$ constant, eight styled convolution stages, and nearest-neighbor upsampling to produce $32 \times 32$ RGB images. After adversarial training, the notebook examines whether different layers appear to control different levels of visual structure by exchanging styles between two latent codes and by traversing a linear path through latent space.

## Research objectives

The notebook supports the following technical questions:

1. Can a small style-based generator learn a useful distribution of low-resolution CIFAR-10 images with a short classroom-scale training run?
2. Does separating the input latent space $Z$ from the intermediate style space $W$ provide an interpretable interface for synthesis?
3. Do early and late style injections produce visibly different effects during style mixing?
4. Does linear interpolation between latent vectors produce a visually continuous sequence of generated images?

## Methodology

### Dataset and preprocessing

The experiment downloads CIFAR-10 through `tf.keras.datasets.cifar10`. It uses the first 20,000 training images to keep the run manageable and ignores the class labels because the generator is unconditional. Each $32 \times 32 \times 3$ image is converted to `float32` and normalized from pixel values in $[0, 255]$ to the generator output range $[-1, 1]$:

$$
x_{normalized} = \frac{x - 127.5}{127.5}.
$$

Images are shuffled with a fixed seed, batched in groups of 64, and prefetched using the TensorFlow data pipeline.

### Generator

The generator consists of four conceptual parts:

1. **Latent mapping network:** an eight-layer multilayer perceptron maps a 128-dimensional latent vector $z$ to a 128-dimensional intermediate vector $w$. The input is normalized before the dense blocks, and each block uses a He-normal initialized dense layer followed by a leaky-ReLU activation.
2. **Learned constant:** synthesis begins from a trainable $4 \times 4 \times 128$ tensor, broadcast across the batch.
3. **Styled convolutions:** eight convolutional blocks process the feature map. The spatial resolution is increased from $4 \times 4$ to $8 \times 8$, $16 \times 16$, and finally $32 \times 32$ between selected blocks.
4. **RGB projection:** a final $1 \times 1$ convolution with `tanh` activation converts the feature map into a three-channel image.

For a feature tensor `x` and style vector `w`, the AdaIN layer computes channel-wise scale and bias parameters. The equation is written in a renderer-safe code block so it remains visible in GitHub and basic Markdown viewers:

```text
AdaIN(x, w) = scale(w) * (x - mean(x)) / sqrt(var(x) + epsilon) + bias(w)
```

where the affine transformation of $w$ produces the scale and bias values. The notebook also adds learned-strength Gaussian noise at each styled convolution, allowing stochastic local variation in the synthesized image.

### Discriminator

The discriminator is a convolutional binary classifier with four downsampling blocks. Its channel widths are 32, 64, 128, and 128. A flattened feature representation is passed to a single linear output (a logit), which is used directly by the softplus adversarial objective.

### Adversarial objective

For a real image `x` and generated image `G(z)`, the discriminator minimizes the non-saturating logistic loss. The generator minimizes the corresponding non-saturating generator loss:

```text
Discriminator: L_D = mean(softplus(-D(real))) + mean(softplus(D(fake)))
Generator:     L_G = mean(softplus(-D(fake)))

where fake = G(z)
```

In these expressions, `D(.)` is the discriminator logit, `G(z)` is a generated image, and `softplus(a) = log(1 + exp(a))`.

Both networks use Adam with learning rate $2 \times 10^{-4}$, $\beta_1 = 0$, and $\beta_2 = 0.99$. The notebook performs one discriminator update and one generator update per batch.

## Experiment configuration

| Parameter | Value |
|---|---:|
| Dataset | CIFAR-10 |
| Training subset | 20,000 images |
| Image resolution | $32 \times 32$ RGB |
| Batch size | 64 |
| Latent dimension $Z$ | 128 |
| Intermediate style dimension $W$ | 128 |
| Styled convolution stages | 8 |
| Training epochs | 10 |
| Optimizer | Adam |
| Learning rate | $2 \times 10^{-4}$ |
| Adam coefficients | $\beta_1=0$, $\beta_2=0.99$ |
| Random seed | 42 |

The values are defined near the beginning of the notebook and can be changed for follow-up experiments.

## Latent-space experiments

### Style mixing

Two latent vectors, $z_A$ and $z_B$, are mapped to $w_A$ and $w_B$. The notebook constructs a sequence of per-layer style tensors by taking the first $k$ styles from $w_A$ and the remaining styles from $w_B$, for cutoffs $k \in \{2, 4, 6\}$. This provides a visual probe of layer-wise style control:

$$
w_i =
\begin{cases}
w_{A,i}, & i < k \\
w_{B,i}, & i \geq k.
\end{cases}
$$

The interpretation that early layers influence coarse structure and later layers influence finer appearance is a hypothesis to inspect visually, not a guaranteed disentanglement result.

### Latent interpolation

The notebook linearly interpolates between two latent vectors over nine steps:

$$
z(t) = (1-t)z_1 + tz_2, \qquad t \in [0,1].
$$

The resulting image strip makes it possible to assess whether the learned generator changes smoothly along this path.

## Installation and execution

### Requirements

- Python 3.9 or newer
- TensorFlow 2.x
- NumPy
- Matplotlib
- Jupyter or Visual Studio Code with the Jupyter extension

Install the dependencies with:

```bash
pip install tensorflow numpy matplotlib jupyter
```

Launch the notebook from the repository root:

```bash
jupyter notebook StyleGAN.ipynb
```

Run the cells from top to bottom. The first dataset-loading cell downloads CIFAR-10 automatically. Training time depends on the available CPU or GPU; ten epochs is a compact demonstration setting rather than a convergence guarantee.

## Outputs and interpretation

The notebook produces:

- A grid of real CIFAR-10 examples for visual reference
- Generator and discriminator parameter counts
- Per-epoch generator and discriminator loss curves
- A grid of generated images after training
- A five-image style-mixing comparison
- A nine-image latent-interpolation sequence

GAN losses alone are not sufficient to establish image quality or diversity. For a formal study, supplement the visual inspection with fixed evaluation protocols such as Fréchet Inception Distance (FID), Kernel Inception Distance (KID), precision/recall for generative models, multiple random seeds, and longer training runs.

## Reproducibility notes

- Python, NumPy, and TensorFlow random seeds are set to `42`.
- The dataset subset and batch construction are deterministic at the notebook level.
- TensorFlow and hardware-specific kernels can still introduce small numerical differences.
- The notebook does not save model checkpoints; rerunning it retrains the networks from initialization.
- The displayed generated samples depend on the trained weights and therefore cannot be reproduced from the README alone.

## Limitations and extensions

This implementation is intentionally small and omits several components required for rigorous comparison with modern StyleGAN systems. In particular, it uses a limited subset of CIFAR-10, trains for only ten epochs, has no checkpointing or experiment tracking, and reports no quantitative generative metrics. The AdaIN mechanism is included to expose the central style-modulation idea, but the architecture should not be treated as an exact implementation of any published StyleGAN version.

Natural next steps include:

- Training on the full dataset for substantially longer
- Adding checkpoint save/restore and configuration logging
- Running multiple seeds and reporting confidence intervals
- Adding FID/KID and precision/recall evaluation
- Comparing direct latent injection against mapped style injection
- Replacing AdaIN with the weight-demodulation approach used in StyleGAN2
- Adding class conditioning to study class-specific synthesis

## Repository structure

```text
.
├── StyleGAN.ipynb   # Complete model, training loop, and latent-space analysis
└── README.md        # Technical and academic project documentation
```

## Academic context

The implementation is informed by the following foundational ideas:

- Goodfellow et al., *Generative Adversarial Nets*, 2014.
- Karras, Laine, and Aila, *A Style-Based Generator Architecture for Generative Adversarial Networks*, 2019.
- Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009, describing CIFAR-10.

This repository is an educational implementation and does not claim to reproduce the reported results of those publications.

## License

No license is currently included. Add an appropriate open-source license before redistributing or accepting external contributions.
