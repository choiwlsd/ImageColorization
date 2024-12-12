# -*- coding: utf-8 -*-
"""MonochromeToColor2.5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iqrZ75x1NVfp8usZd5U8_UBBRrYSt7F7
"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("shravankumar9892/image-colorization")

print("Path to dataset files:", path)

!pip install torchsummary

!pip install pytorch-lightning

# Commented out IPython magic to ensure Python compatibility.
import os
from pathlib import Path

# Import glob to get the files directories recursively
import glob

# Import Garbage collector interface
import gc

# Import OpenCV to transforme pictures
import cv2

# Import Time
import time

# import numpy for math calculations
import numpy as np

# Import pandas for data (csv) manipulation
import pandas as pd

# Import matplotlib for plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('fivethirtyeight')
# %matplotlib inline

import PIL
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

import pytorch_lightning as pl

# Import pytorch to build Deel Learling Models
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
from scipy.stats import entropy

from torchsummary import summary

# Import tqdm to show a smart progress meter
from tqdm import tqdm

# Import warnings to hide the unnessairy warniings
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path 변수에 저장된 디렉토리 내 파일 목록 출력
print("Dataset contents:")
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))

# npy 파일 열기
gray_scale_path = f"{path}/l/gray_scale.npy"
ab1_path = f"{path}/ab/ab/ab1.npy"

# 데이터를 로드
L_df = (np.load(gray_scale_path))[:500]
ab1_df = (np.load(ab1_path))[:500]

# 데이터 확인
print("Gray Scale Shape:", L_df.shape) # (h, w) 형식
print("AB1 Shape:", ab1_df.shape) # (h, w, 2) 형식

print("AB1 min:", ab1_df.min(), "max:", ab1_df.max())

L_df.min(), L_df.max()

dataset = (L_df, ab1_df)
gc.collect()

def lab_to_rgb(L, ab):
    """
    Takes an image or a batch of images and converts from LAB space to RGB
    """
    L = L  * 100
    ab = (ab - 0.5) * 128 * 2
    Lab = torch.cat([L, ab], dim=2).numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

plt.figure(figsize=(30,30))
for i in range(1,16,2):
    plt.subplot(4,4,i)
    img = np.zeros((224,224,3))
    img[:,:,0] = L_df[i]
    plt.title('B&W')
    plt.imshow(lab2rgb(img))

    plt.subplot(4,4,i+1)
    img[:,:,1:] = ab1_df[i]
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    plt.title('Colored')
    plt.imshow(img)

gc.collect()



"""## Data Loader"""

class ImageColorizationDataset(Dataset):
    ''' Black and White (L) Images and corresponding A&B Colors'''
    def __init__(self, dataset, transform=None):
        '''
        :param dataset: Dataset name.
        :param data_dir: Directory with all the images.
        :param transform: Optional transform to be applied on sample
        '''
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        L = np.array(dataset[0][idx]).reshape((224,224,1))
        L = transforms.ToTensor()(L)

        ab = np.array(dataset[1][idx])
        ab = transforms.ToTensor()(ab)

        return ab, L

batch_size = 1

# Prepare the Datasets
train_dataset = ImageColorizationDataset(dataset = (L_df, ab1_df))
test_dataset = ImageColorizationDataset(dataset = (L_df, ab1_df))

# Build DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True, pin_memory = True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle = False, pin_memory = True)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.identity_map = nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=stride)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputs):
        x = inputs.clone().detach()
        out = self.layer(x)
        residual  = self.identity_map(inputs)
        skip = out + residual
        return self.relu(skip)

class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(in_channels, out_channels)
        )

    def forward(self, inputs):
        return self.layer(inputs)

class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res_block = ResBlock(in_channels + out_channels, out_channels)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_channel, output_channel, dropout_rate = 0.2):
        super().__init__()
        self.encoding_layer1_ = ResBlock(input_channel,64)
        self.encoding_layer2_ = DownSampleConv(64, 128)
        self.encoding_layer3_ = DownSampleConv(128, 256)
        self.bridge = DownSampleConv(256, 512)
        self.decoding_layer3_ = UpSampleConv(512, 256)
        self.decoding_layer2_ = UpSampleConv(256, 128)
        self.decoding_layer1_ = UpSampleConv(128, 64)
        self.output = nn.Conv2d(64, output_channel, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, inputs):
        ###################### Enocoder #########################
        e1 = self.encoding_layer1_(inputs)
        e1 = self.dropout(e1)
        e2 = self.encoding_layer2_(e1)
        e2 = self.dropout(e2)
        e3 = self.encoding_layer3_(e2)
        e3 = self.dropout(e3)

        ###################### Bridge #########################
        bridge = self.bridge(e3)
        bridge = self.dropout(bridge)

        ###################### Decoder #########################
        d3 = self.decoding_layer3_(bridge, e3)
        d2 = self.decoding_layer2_(d3, e2)
        d1 = self.decoding_layer1_(d2, e1)

        ###################### Output #########################
        output = self.output(d1)
        return output

model = Generator(1,2).to(device)
summary(model, (1, 224, 224), batch_size = 1)

class Critic(nn.Module):
    def __init__(self, in_channels=3):
        super(Critic, self).__init__()

        def critic_block(in_filters, out_filters, normalization=True):
            """Returns layers of each critic block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *critic_block(in_channels, 64, normalization=False),
            *critic_block(64, 128),
            *critic_block(128, 256),
            *critic_block(256, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, ab, l):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((ab, l), 1)
        output = self.model(img_input)
        return output

model = Critic(3).to(device)
summary(model, [(2, 224, 224), (1, 224, 224)], batch_size = 1)

# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def display_progress(cond, real, fake, current_epoch = 0, figsize=(20,15)):
    """
    Save cond, real (original) and generated (fake)
    images in one panel
    """
    cond = cond.detach().cpu().permute(1, 2, 0)
    real = real.detach().cpu().permute(1, 2, 0)
    fake = fake.detach().cpu().permute(1, 2, 0)

    images = [cond, real, fake]
    titles = ['input','real','generated']
    print(f'Epoch: {current_epoch}')
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for idx,img in enumerate(images):
        if idx == 0:
            ab = torch.zeros((224,224,2))
            img = torch.cat([images[0]* 100, ab], dim=2).numpy()
            imgan = lab2rgb(img)
        else:
            imgan = lab_to_rgb(images[0],img)
        ax[idx].imshow(imgan)
        ax[idx].axis("off")
    for idx, title in enumerate(titles):
        ax[idx].set_title('{}'.format(title))
    plt.show()

class CWGAN(pl.LightningModule):

    def __init__(self, in_channels, out_channels, learning_rate=0.0002, lambda_recon=100, display_step=10, lambda_gp=10, lambda_r1=10, step_lr_gamma=0.95, step_lr_step_size=10):
        super().__init__()
        self.automatic_optimization = False

        self.save_hyperparameters()

        self.display_step = display_step

        self.generator = Generator(in_channels, out_channels)
        self.critic = Critic(in_channels + out_channels)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.optimizer_C = optim.Adam(self.critic.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.lambda_recon = lambda_recon
        self.lambda_gp = lambda_gp
        self.lambda_r1 = lambda_r1
        self.recon_criterion = nn.L1Loss()
        self.generator_losses, self.critic_losses = [], []

        # Step LR Scheduler parameters
        self.step_lr_gamma = step_lr_gamma
        self.step_lr_step_size = step_lr_step_size

    def configure_optimizers(self):
        scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=self.step_lr_step_size, gamma=self.step_lr_gamma)
        scheduler_C = optim.lr_scheduler.StepLR(self.optimizer_C, step_size=self.step_lr_step_size, gamma=self.step_lr_gamma)
        return [self.optimizer_C, self.optimizer_G], [scheduler_C, scheduler_G]

    def generator_step(self, real_images, conditioned_images):
        # WGAN has only a reconstruction loss
        self.optimizer_G.zero_grad()
        fake_images = self.generator(conditioned_images)
        recon_loss = self.recon_criterion(fake_images, real_images)
        recon_loss.backward()
        self.optimizer_G.step()

        # Keep track of the average generator loss
        self.generator_losses += [recon_loss.item()]

    def critic_step(self, real_images, conditioned_images):
        self.optimizer_C.zero_grad()
        fake_images = self.generator(conditioned_images)
        fake_logits = self.critic(fake_images, conditioned_images)
        real_logits = self.critic(real_images, conditioned_images)

        # Compute the loss for the critic
        loss_C = real_logits.mean() - fake_logits.mean()

        # Compute the gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1, requires_grad=True)
        alpha = alpha.to(device)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images.detach()).requires_grad_(True)

        interpolated_logits = self.critic(interpolated, conditioned_images)

        grad_outputs = torch.ones_like(interpolated_logits, dtype=torch.float32, requires_grad=True)
        gradients = torch.autograd.grad(outputs=interpolated_logits, inputs=interpolated, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(len(gradients), -1)
        gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        loss_C += self.lambda_gp * gradients_penalty

        # Compute the R1 regularization loss
        r1_reg = gradients.pow(2).sum(1).mean()
        loss_C += self.lambda_r1 * r1_reg

        # Backpropagation
        loss_C.backward()
        self.optimizer_C.step()
        self.critic_losses += [loss_C.item()]

    def training_step(self, batch, batch_idx):
        real, condition = batch
        # 두 Optimizer 가져오기
        opt_critic, opt_generator = self.optimizers()
        scheduler_critic, scheduler_generator = self.lr_schedulers()

        # Critic 업데이트 단계
        if batch_idx % 5 != 0:  # Critic을 5번 더 자주 업데이트
            self.critic_step(real, condition)
            opt_critic.step()
            opt_critic.zero_grad()

        # Generator 업데이트 단계
        else:
            self.generator_step(real, condition)
            opt_generator.step()
            opt_generator.zero_grad()

        # Loss 계산 및 기록
        gen_mean = sum(self.generator_losses[-self.display_step:]) / self.display_step
        crit_mean = sum(self.critic_losses[-self.display_step:]) / self.display_step

        # Scheduler step
        if batch_idx == 0:  # Perform scheduler step at the end of every epoch
            scheduler_critic.step()
            scheduler_generator.step()

        # Epoch마다 상태 저장 및 출력
        if self.current_epoch % self.display_step == 0 and batch_idx == 0:
            fake = self.generator(condition).detach()
            torch.save(self.generator.state_dict(), f"ResUnet_{self.current_epoch}.pt")
            torch.save(self.critic.state_dict(), f"PatchGAN_{self.current_epoch}.pt")
            print(f"Epoch {self.current_epoch} : Generator loss: {gen_mean}, Critic loss: {crit_mean}")
            display_progress(condition[0], real[0], fake[0], self.current_epoch)

gc.collect()
cwgan = CWGAN(in_channels = 1, out_channels = 2, lambda_recon=75, lambda_gp = 15, display_step=10)

trainer = pl.Trainer(max_epochs=150, accelerator='gpu', devices=-1)
trainer.fit(cwgan, train_loader)

