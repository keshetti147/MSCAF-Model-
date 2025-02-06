# MSCAF-Model-
Implementation
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from einops import rearrange
from timm.models.swin_transformer import SwinTransformer
import cv2
import numpy as np
from PIL import Image

# ----------------------
# Data Preprocessing
# ----------------------
class MedicalImageDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ----------------------
# Feature Extraction
# ----------------------
class ModifiedVGG19(nn.Module):
    def __init__(self):
        super(ModifiedVGG19, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features[:36]  # Extracting Features Only
        self.gelu = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.vgg19(x)
        x = self.gelu(x)
        x = self.pool(x)
        return x

class AttCNN(nn.Module):
    def __init__(self):
        super(AttCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128 // 16, 1),
            nn.ReLU(),
            nn.Conv2d(128 // 16, 128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.pool(x)
        se_weight = self.se(x)
        x = x * se_weight
        return x

# ----------------------
# Fusion using Swin Transformer & Cross Attention
# ----------------------
class SwinCrossAttentionFusion(nn.Module):
    def __init__(self):
        super(SwinCrossAttentionFusion, self).__init__()
        self.swin_transformer = SwinTransformer(img_size=224, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2],
                                               num_heads=[3, 6, 12, 24], window_size=7, num_classes=0)
        self.cross_attention = nn.MultiheadAttention(embed_dim=96, num_heads=4)
    
    def forward(self, ct_features, mri_features):
        fused_features = self.swin_transformer(ct_features) + self.swin_transformer(mri_features)
        fused_features = rearrange(fused_features, 'b c h w -> (h w) b c')
        fused_features, _ = self.cross_attention(fused_features, fused_features, fused_features)
        fused_features = rearrange(fused_features, '(h w) b c -> b c h w', h=14, w=14)
        return fused_features

# ----------------------
# Image Reconstruction
# ----------------------
class ImageReconstruction(nn.Module):
    def __init__(self):
        super(ImageReconstruction, self).__init__()
        self.conv1 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x

# ----------------------
# MSCAF Model Pipeline
# ----------------------
class MSCAF(nn.Module):
    def __init__(self):
        super(MSCAF, self).__init__()
        self.feature_extractor_ct = ModifiedVGG19()
        self.feature_extractor_mri = AttCNN()
        self.fusion_model = SwinCrossAttentionFusion()
        self.reconstruction_model = ImageReconstruction()
    
    def forward(self, ct_img, mri_img):
        ct_features = self.feature_extractor_ct(ct_img)
        mri_features = self.feature_extractor_mri(mri_img)
        fused_features = self.fusion_model(ct_features, mri_features)
        output = self.reconstruction_model(fused_features)
        return output

# ----------------------
# Model Training
# ----------------------
def train_model():
    model = MSCAF().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    dataset = MedicalImageDataset(["path_to_ct_image", "path_to_mri_image"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(10):
        for batch in dataloader:
            ct_img = batch.cuda()
            mri_img = batch.cuda()
            optimizer.zero_grad()
            fused_output = model(ct_img, mri_img)
            loss = criterion(fused_output, ct_img)  # Fusion output should resemble CT images
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    train_model()
