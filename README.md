# StyleGAN Exploration

A compact TensorFlow/Keras exploration of StyleGAN-inspired image generation on CIFAR-10. The notebook implements a mapping network, adaptive instance normalization, noise injection, a progressive style-based generator, and a convolutional discriminator.

## What is included

- Deterministic setup with seed `42`
- CIFAR-10 preprocessing using a 20,000-image training subset
- An 8-layer latent mapping network from `z` to `w`
- Learned constant input and styled convolution blocks
- AdaIN-based style modulation and per-layer noise injection
- Non-saturating softplus GAN losses with Adam optimizers
- Generated sample grids and generator/discriminator loss curves
- Style mixing at multiple layer cutoffs
- Linear latent-space interpolation between two samples

## Requirements

- Python 3.9+
- TensorFlow 2.x
- NumPy
- Matplotlib

Install the dependencies with:

```bash
pip install tensorflow numpy matplotlib jupyter
```

## Run

Open `StyleGAN.ipynb` in Jupyter or Visual Studio Code and run the cells from top to bottom:

```bash
jupyter notebook StyleGAN.ipynb
```

The first run downloads CIFAR-10 automatically. Training is configured for 10 epochs with batch size 64 and may take several minutes or longer depending on available CPU/GPU hardware.

## Configuration

The main experiment settings are defined near the start of the notebook:

```python
TRAIN_LIMIT = 20000
BATCH = 64
EPOCHS = 10
Z_DIM = W_DIM = 128
```

Increase `TRAIN_LIMIT` or `EPOCHS` for a longer experiment. The notebook is intended for educational exploration rather than a production-quality StyleGAN reproduction.

## Project structure

```text
.
├── StyleGAN.ipynb   # Model, training loop, and latent-space experiments
└── README.md        # Project documentation
```

## License

This repository does not currently include a license. Add one before distributing the project for reuse.
