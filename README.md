# Deep Image Prior with Denoising Diffusion Probabilistic Models (DDPM) on CIFAR-10

## Overview

This project aims to compare the performance of Deep Image Prior (DIP) with and without the integration of Denoising Diffusion Probabilistic Models (DDPM) on the CIFAR-10 dataset for image reconstruction. The goal is to evaluate the impact of DDPM on image quality metrics such as Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).

## Setup

### Prerequisites

- Python 3.7+
- CUDA (for GPU support, optional but recommended)

### Installation

Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Experiment
To run the experiment, execute the following command:
```bash
python main.py
```
This script will train the DIP models with and without DDPM on the CIFAR-10 dataset and plot the results comparing the two approaches.
   
## Results
The results will include plots comparing the loss, PSNR, and SSIM metrics for the models with and without DDPM. Additionally, reconstructed images will be saved for visual comparison.

## Reproducibility
To ensure the reproducibility of the results, random seeds have been set for all libraries involved in the randomization process.
