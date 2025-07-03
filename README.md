## GAN on MNIST

This repository contains an implementation of a basic Generative Adversarial Network (GAN) trained on the MNIST dataset using PyTorch.

---

### Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Training Process](#training-process)
* [Logging & Visualization](#logging--visualization)
* [Results](#results)
* [Directory Structure](#directory-structure)
* [Requirements](#requirements)
* [License](#license)

---

### Overview

This project implements a simple fully-connected GAN to generate handwritten digit images similar to MNIST. The generator maps random noise vectors to image space and the discriminator learns to distinguish real MNIST images from generated ones.

### Features

* **Configurable** via `GANConfig` dataclass
* **Training Loop** with separate generator and discriminator updates
* **Weight Initialization** following DCGAN guidelines
* **TensorBoard Logging** for generator and discriminator losses
* **Visualization** of generated and real image grids during training

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/gan-mnist.git
cd gan-mnist
```

2. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

All hyperparameters and file paths can be configured via the `GANConfig` dataclass in `train.py`:

* `data_dir`: Path to download MNIST data
* `output_dir`: Directory to save model checkpoints and TensorBoard logs
* `batch_size`: Mini-batch size (default: 128)
* `z_dim`: Dimensionality of the noise vector (default: 100)
* `lr`: Learning rate for Adam optimizer (default: 2e-4)
* `betas`: Beta parameters for Adam optimizer (default: (0.5, 0.999))
* `epochs`: Number of training epochs (default: 100)
* `img_size`: Size to resize MNIST images (default: 28)
* `log_interval`: Steps between logging and visualization (default: 300)

### Usage

To start training:

```bash
python train.py --data_dir ./data --output_dir ./outputs --batch_size 128 --z_dim 100 --lr 0.0002 --epochs 50
```

TensorBoard logs will be saved in the `outputs/logs` directory:

```bash
tensorboard --logdir outputs/logs
```

### Model Architecture

* **Generator**: Fully-connected network with layers \[100 → 256 → 512 → 1024 → 784], using ReLU activations and BatchNorm, output using Tanh.
* **Discriminator**: Fully-connected network with layers \[784 → 512 → 256 → 1], using LeakyReLU activations.

### Training Process

1. Initialize generator and discriminator with normal weight init.
2. For each batch:

   * Update discriminator on real and fake images.
   * Update generator to fool the discriminator.
3. Log losses and periodically visualize sample images.

### Logging & Visualization

* Loss curves for generator and discriminator saved to TensorBoard.
* Generated and real image grids plotted every `log_interval` steps.


### Directory Structure

```
├── data/               # MNIST data
├── outputs/            # Checkpoints and logs
│   ├── logs/           # TensorBoard logs
│   ├── generator.pth
│   └── discriminator.pth
├── train.py            # Training script
├── requirements.txt    # Python dependencies
└── README.md           # Project overview
```

### Requirements

* Python 3.7+
* PyTorch
* torchvision
* tensorboard
* matplotlib
* tqdm

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
