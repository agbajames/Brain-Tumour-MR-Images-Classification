# Brain Tumor MRI Classification

A comprehensive deep learning project utilizing EfficientNet architectures to classify brain tumor types from MRI images. This project systematically evaluates multiple EfficientNet variants (B0-B7) across different training configurations to identify optimal classification strategies.

## 🎯 Project Overview

This MSc thesis project focuses on automated brain tumor classification using state-of-the-art convolutional neural network architectures. The system classifies MRI scans into three tumor types:
- **Glioma**
- **Meningioma**
- **Pituitary**

## 🏆 Key Results

| Model | Configuration | Test Accuracy |
|-------|--------------|---------------|
| **EfficientNetB5** | 80/20 + Augmentation | **99.67%** |
| **EfficientNetB7** | 80/20 + Augmentation | **99.67%** |
| **EfficientNetB2** | 80/20 + Augmentation | 99.35% |
| EfficientNetB4 | 80/20 + Augmentation | 99.35% |
| EfficientNetB1 | 70/30 + Augmentation | 99.24% |

### Performance by Configuration

**80/20 Split with Augmentation** (Best Overall)
- EfficientNetB5: 99.67%
- EfficientNetB7: 99.67%
- EfficientNetB2: 99.35%
- EfficientNetB4: 99.35%

**70/30 Split with Augmentation**
- EfficientNetB1: 99.24%
- EfficientNetB4: 99.13%
- EfficientNetB0: 99.02%

**80/20 Split without Augmentation**
- EfficientNetB4: 98.80%
- EfficientNetB1: 98.48%
- EfficientNetB0: 98.37%

## 🛠️ Technical Stack

- **Framework**: TensorFlow 2.9.0 / Keras
- **Architecture**: EfficientNet (B0 through B7)
- **Transfer Learning**: ImageNet pretrained weights
- **Image Processing**: OpenCV, PIL
- **Data Handling**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn

## 📊 Dataset

The dataset consists of brain MRI images organized into three tumor categories:
- Training set sizes varied between configurations (70/30 and 80/20 splits)
- Image preprocessing: Resized to 240x240 pixels
- Data augmentation: Brightness adjustments for improved generalization

**Training Data (70/30 Split with Augmentation)**:
- Glioma: 4,989 images
- Meningioma: 4,452 images  
- Pituitary: 4,555 images

## 🔬 Methodology

### Model Architecture
- **Base Model**: EfficientNet variants (B0-B7) pretrained on ImageNet
- **Transfer Learning**: Feature extraction from pre-trained weights
- **Fine-tuning**: Custom classification head for 3-class tumor classification
- **Image Size**: 240x240x3

### Training Configuration
- **Optimizer**: Adam with learning rate scheduling
- **Learning Rate**: Initial 0.001, reduced on plateau
- **Callbacks**: 
  - Early Stopping
  - ReduceLROnPlateau
  - ModelCheckpoint
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

### Data Augmentation
Augmentation techniques applied to improve model generalization:
- Brightness variations
- Image normalization
- Random transformations during training

## 📁 Project Structure

```
Brain-Tumour-MR-Images-Classification/
├── Converting_Image_from_MAT_to_JPG.ipynb
├── Crop_images.ipynb
├── split_images.ipynb
├── EffNetModels_70_30_Split_With_Agument...
├── EffNetModels_70_30_Split_Without_Agum...
├── EfficientNet_Models_80_20_Split_With_A...
└── README.md
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow==2.9.0
pip install opencv-python
pip install numpy pandas
pip install matplotlib seaborn
pip install scikit-learn
```

### Usage

1. **Data Preparation**
   - Run `Converting_Image_from_MAT_to_JPG.ipynb` to convert raw data
   - Execute `Crop_images.ipynb` for image preprocessing
   - Use `split_images.ipynb` to create train/test splits

2. **Model Training**
   - Choose appropriate notebook based on desired configuration
   - Notebooks follow naming convention: `EffNetModels_{split}_{augmentation}`

3. **Evaluation**
   - Each notebook includes comprehensive evaluation metrics
   - Confusion matrices and classification reports generated
   - Test accuracy computed on held-out test set

## 📈 Key Findings

1. **Augmentation Impact**: Data augmentation consistently improved model performance across all architectures, with gains of 0.5-1.5% in test accuracy

2. **Model Scaling**: Larger EfficientNet variants (B5, B7) achieved the highest accuracy (99.67%), though smaller models (B0, B1) still achieved >99% with proper augmentation

3. **Split Ratio**: The 80/20 split with augmentation produced the best results, balancing training data volume with evaluation reliability

4. **Convergence**: All models achieved validation accuracy >99% within 30-40 epochs with proper learning rate scheduling

## 🎓 Academic Context

This project was completed as part of an MSc thesis, demonstrating practical application of:
- Transfer learning with state-of-the-art CNN architectures
- Systematic hyperparameter evaluation
- Medical image classification techniques
- Experimental design and result analysis

## 📝 Citation

If you use this work in your research, please cite:
```
Brain Tumor Classification using EfficientNet Architectures
MSc Thesis Project, 2022
```

## 🔗 Future Work

Potential improvements and extensions:
- Multi-modal fusion (combining different MRI sequences)
- Explainability visualization (Grad-CAM, attention maps)
- Deployment as REST API for clinical integration
- Extended dataset with additional tumor types
- Cross-validation for more robust performance estimates

## 📜 License

This project is available under the MIT License - see LICENSE file for details.

## 🤝 Contributing

While this is a completed thesis project, suggestions and discussions are welcome. Please open an issue for any questions or recommendations.

## ⚠️ Disclaimer

This project is for research and educational purposes only. It should not be used for clinical diagnosis without proper validation and regulatory approval.

---

**Author**: James Agba  
**Year**: 2022  
**Institution**: University of Hull  
