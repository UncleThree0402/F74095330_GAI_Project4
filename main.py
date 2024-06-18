import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from DIP import DIP
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.transforms import ToPILImage
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# DDPM Setting
unet_model = Unet(
    dim=64,
    dim_mults=(1, 2, 4)
)
unet_model.to(device)

gaussian_diffusion = GaussianDiffusion(
    unet_model,
    image_size=32,
    timesteps=1000,
)
gaussian_diffusion.to(device)

# Transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
subset_indices = indices[:int(0.05 * num_train)]

subset_train_dataset = Subset(train_dataset, subset_indices)
train_loader = DataLoader(subset_train_dataset, batch_size=32, shuffle=True)

gaussian_diffusion_list = [0.1, 0.3, 0.5]

# With DDPM
dip_model = DIP()
dip_model = dip_model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(dip_model.parameters(), lr=0.001)

ddpm_loss = []
ddpm_psnr = []
ddpm_ssim = []

epoch = 50
for epoch in range(epoch):
    train_loss = []
    train_psnr = []
    train_ssim = []

    for images, _ in train_loader:
        images = images.to(device)
        for gaussian_level in gaussian_diffusion_list:
            gaussian_images = gaussian_diffusion.q_sample(images, torch.tensor(
                [int(gaussian_level * (gaussian_diffusion.num_timesteps - 1))], device=device).long())
            optimizer.zero_grad()
            gaussian_images_outputs = dip_model(gaussian_images)
            loss = criterion(gaussian_images_outputs, images)
            loss.backward()
            optimizer.step()

            images_np = images.cpu().numpy()
            gaussian_images_outputs_np = gaussian_images_outputs.detach().cpu().numpy()
            psnr_result = psnr(images_np, gaussian_images_outputs_np, data_range=1.0)
            ssim_result = ssim(images_np, gaussian_images_outputs_np, multichannel=True, win_size=3, data_range=1.0,
                               channel_axis=-1)

            train_loss.append(loss.item())
            train_psnr.append(psnr_result)
            train_ssim.append(ssim_result)

    avg_loss = np.mean(train_loss)
    avg_psnr = np.mean(train_psnr)
    avg_ssim = np.mean(train_ssim)
    ddpm_loss.append(avg_loss)
    ddpm_psnr.append(avg_psnr)
    ddpm_ssim.append(avg_ssim)
    print(
        f"Epoch {epoch + 1} Average Loss {avg_loss:.4f} Average psnr {avg_psnr:.4f} Average ssim {avg_ssim:.4f}")

# No DDPM
no_ddpm_dip_model = DIP()
no_ddpm_dip_model = no_ddpm_dip_model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(no_ddpm_dip_model.parameters(), lr=0.001)

no_ddpm_loss = []
no_ddpm_psnr = []
no_ddpm_ssim = []

epoch = 50
for epoch in range(epoch):
    train_loss = []
    train_psnr = []
    train_ssim = []

    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        images_outputs = no_ddpm_dip_model(images)
        loss = criterion(images_outputs, images)
        loss.backward()
        optimizer.step()

        images_np = images.cpu().numpy()
        images_outputs_np = images_outputs.detach().cpu().numpy()
        psnr_result = psnr(images_np, images_outputs_np, data_range=1.0)
        ssim_result = ssim(images_np, images_outputs_np, multichannel=True, win_size=3, data_range=1.0, channel_axis=-1)

        train_loss.append(loss.item())
        train_psnr.append(psnr_result)
        train_ssim.append(ssim_result)

    avg_loss = np.mean(train_loss)
    avg_psnr = np.mean(train_psnr)
    avg_ssim = np.mean(train_ssim)
    no_ddpm_loss.append(avg_loss)
    no_ddpm_psnr.append(avg_psnr)
    no_ddpm_ssim.append(avg_ssim)
    print(
        f"Epoch {epoch + 1} Average Loss {avg_loss:.4f} Average psnr {avg_psnr:.4f} Average ssim {avg_ssim:.4f}")

# Plot difference

# Calculate differences
loss_diff = np.array(ddpm_loss) - np.array(no_ddpm_loss)
psnr_diff = np.array(ddpm_psnr) - np.array(no_ddpm_psnr)
ssim_diff = np.array(ddpm_ssim) - np.array(no_ddpm_ssim)

# Plotting
epochs = range(1, len(ddpm_loss) + 1)

plt.figure(figsize=(18, 12))

# Plot ddpm_loss vs no_ddpm_loss
plt.subplot(2, 3, 1)
sns.lineplot(x=epochs, y=ddpm_loss, label='DDPM Loss')
sns.lineplot(x=epochs, y=no_ddpm_loss, label='No DDPM Loss')
plt.title('DDPM Loss vs No DDPM Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot ddpm_psnr vs no_ddpm_psnr
plt.subplot(2, 3, 2)
sns.lineplot(x=epochs, y=ddpm_psnr, label='DDPM PSNR')
sns.lineplot(x=epochs, y=no_ddpm_psnr, label='No DDPM PSNR')
plt.title('DDPM PSNR vs No DDPM PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()

# Plot ddpm_ssim vs no_ddpm_ssim
plt.subplot(2, 3, 3)
sns.lineplot(x=epochs, y=ddpm_ssim, label='DDPM SSIM')
sns.lineplot(x=epochs, y=no_ddpm_ssim, label='No DDPM SSIM')
plt.title('DDPM SSIM vs No DDPM SSIM')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.legend()

# Plot difference in loss
plt.subplot(2, 3, 4)
sns.lineplot(x=epochs, y=loss_diff, label='Loss Difference (DDPM - No DDPM)')
plt.title('Difference in Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Difference')
plt.legend()

# Plot difference in PSNR
plt.subplot(2, 3, 5)
sns.lineplot(x=epochs, y=psnr_diff, label='PSNR Difference (DDPM - No DDPM)')
plt.title('Difference in PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR Difference')
plt.legend()

# Plot difference in SSIM
plt.subplot(2, 3, 6)
sns.lineplot(x=epochs, y=ssim_diff, label='SSIM Difference (DDPM - No DDPM)')
plt.title('Difference in SSIM')
plt.xlabel('Epoch')
plt.ylabel('SSIM Difference')
plt.legend()

plt.suptitle('Comparison of DDPM and No DDPM Metrics (Train With Gaussian level mix)', fontsize=16)
plt.tight_layout()
plt.savefig("compare_train_level_mix.png")
plt.show()

dip_model.eval()
no_ddpm_dip_model.eval()
with torch.no_grad():
    for images, _ in train_loader:
        for image in images:
            image = image.to(device)
            for gaussian_level in [0.1, 0.3, 0.5, 0.7, 1]:
                gaussian_image = gaussian_diffusion.q_sample(image, torch.tensor(
                    [int(gaussian_level * (gaussian_diffusion.num_timesteps - 1))], device=device).long())

                dip_model_outputs = dip_model(gaussian_image)
                no_ddpm_dip_model_outputs = no_ddpm_dip_model(gaussian_image)
                PIL_image = ToPILImage()(image.cpu())
                PIL_gaussian_image = ToPILImage()(gaussian_image.cpu())

                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                axs[0].imshow(PIL_image)
                axs[0].set_title('Original Image')
                axs[1].imshow(PIL_gaussian_image)
                axs[1].set_title(f'Gaussian Image ({gaussian_level})')
                axs[2].imshow(ToPILImage()(dip_model_outputs.cpu()))
                axs[2].set_title('Reconstructed Image DIP with DDPM')
                axs[3].imshow(ToPILImage()(no_ddpm_dip_model_outputs.cpu()))
                axs[3].set_title('Reconstructed Image DIP without DDPM')
                plt.savefig(f"image_train_level_mix_{gaussian_level}.png")
                plt.show()

            break
        break
