# MSCAF-Fusion

This repository contains the **Modified Swin-based Cross Attention Fusion (MSCAF)** model for **multimodal medical image fusion**.

## 📜 Features
- **Modified VGG19** for CT/PET feature extraction
- **Attention-based CNN (AttCNN)** for MRI feature extraction
- **Swin Transformer + Cross Attention** for multimodal feature fusion
- **CNN-based image reconstruction** for high-quality output
- **PyTorch implementation with training script**

## 🚀 Installation
Ensure you have Python 3.6+ and install the dependencies:

```bash
pip install torch torchvision timm einops numpy opencv-python
```

## 📌 Usage
To train the MSCAF model, run the following command:

```bash
python mscaf_fusion.py
```

## 📊 Results
The proposed MSCAF model achieves superior performance in multimodal medical image fusion. Below are key evaluation metrics:

| Metric | Value |
|--------|-------|
| SSIM   | 0.989 |
| PSNR   | 52.5  |
| Fusion Factor (FF) | 8.93 |
| Entropy (EN) | 6.12 |

## 📂 Dataset
The datasets used for training and testing the MSCAF model:

1. **CT and MRI Brain Scans Dataset**  
   - Source: Kaggle  
   - URL: [CT-to-MRI Dataset](https://www.kaggle.com/datasets/darren2020/ct-to-mri-cgan)  
   - Images: 5005 (CT: 2486, MRI: 2488)  

2. **Brain MRI and PET Images Dataset**  
   - Source: Harvard Medical School  
   - URL: [Brain MRI-PET Dataset](https://www.med.harvard.edu/AANLIB/home.html)  
   - Images: 60 (MRI: 30, PET: 30)  

## 🔗 Reference
[Paper Link (if applicable)](https://yourpaperlink.com)

## 🤝 Contribution
Feel free to fork this repository and submit a pull request to improve the model!

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
