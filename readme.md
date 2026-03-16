[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/DtxdB3_i)
# Food-11 Image Classification with Transfer Learning
 
## Project Overview
 
This project applies **Transfer Learning** using **EfficientNetB0** to classify food images into 11 categories (Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles-Pasta, Rice, Seafood, Soup, Vegetable-Fruit) using the Food-11 dataset from Kaggle.
 
Four transfer learning strategies were compared, all tracked with **MLflow** on **DAGsHub**.
 
## Environment
 
| Component | Version |
|-----------|---------|
| Model | EfficientNetB0 |
| TensorFlow | 2.19.0 |
| Keras | 3.13.2 |
| GPU | NVIDIA T4 (Google Colab) |
| Image Size | 224 × 224 |
| Batch Size | 32 |
 
## Dataset
 
| Split | Images |
|-------|--------|
| Training | 9,866 |
| Validation | 3,430 |
| Test | 3,347 |
 
Preprocessing: Data augmentation (random horizontal flip, rotation ±36°, zoom ±10%), images resized to 224×224, pipeline optimized with `cache()`, `shuffle()`, and `prefetch()`.
 
---
 
## Experiment Summary
 
| # | Strategy | Test Accuracy | Test Loss | Epochs | Trainable Params |
|---|----------|:---:|:---:|:---:|:---:|
| 1 | Feature Extraction | 89.60% | 0.3124 | 10 | 14,091 |
| 2 | Fine-tuning (Last 20 layers) | 91.22% | 0.2695 | 20 | 1,356,859 |
| 3 | **Gradual Unfreezing** | **94.23%** | **0.2091** | **40** | **182 layers** |
| 4 | Layer-wise LR Decay | 86.32% | 0.4268 | 11 (early stop) | 115 weights |
 
---
 
## Strategy Details
 
### 1. Feature Extraction
- Froze entire EfficientNetB0 backbone
- Trained only the classification head: GlobalAveragePooling2D → Dropout(0.2) → Dense(11, softmax)
- Optimizer: Adam (LR = 1e-3)
- Callbacks: EarlyStopping (patience=5), ReduceLROnPlateau
 
### 2. Fine-tuning (Last N Layers)
- Continued from Feature Extraction model (head already trained)
- Unfroze last 20 layers, kept BatchNorm frozen
- Optimizer: Adam (LR = 1e-5, 100× smaller than Feature Extraction)
- Trained epochs 11–20
 
### 3. Gradual Unfreezing
- Built a fresh model, unfroze one block at a time: top → block7 → block6 → ... → block1
- 5 epochs per block, LR decayed by 0.7× after each block
- BatchNorm frozen throughout
- All blocks kept their learned features while adapting progressively
 
### 4. Layer-wise Learning Rate Decay (Gradient Scaling)
- All layers open from epoch 1
- Used a single Adam optimizer with gradient scaling: each block's gradients multiplied by a decay factor (head ×1.0, top ×0.5, block7 ×0.25, ... stem ×0.002)
- This makes deeper layers learn slower without needing multiple optimizers
 
---
 
## Metrics and Plots
 
### Training Curves
 
Each strategy's training was plotted with Loss and Accuracy curves for both training and validation sets:
 
- **Feature Extraction**: Smooth convergence over 10 epochs, train/val curves close together
- **Fine-tuning**: Continued improvement from epoch 11–20, slight train/val gap
- **Gradual Unfreezing**: 40 epochs with vertical lines marking each block transition
- **Layer-wise LR**: Noisy validation curve, stopped at epoch 11
 
### Final Comparison (Bar Chart)
 
A side-by-side bar chart comparing test accuracy across all 4 strategies shows Gradual Unfreezing as the clear winner at 94.23%.
 
All plots are saved as PNG artifacts in MLflow runs on DAGsHub.
 
---
 
## Observations
 
### Feature Extraction vs Fine-tuning
 
Feature Extraction achieved 89.60% by training only 14K parameters (the head), while Fine-tuning reached 91.22% by additionally training 1.3M parameters in the last 20 layers. The +1.61% improvement confirms that adapting the backbone's higher-level features to food-specific patterns provides meaningful gains. The key was using a much smaller learning rate (1e-5 vs 1e-3) to avoid destroying pretrained weights.
 
### Convergence
 
- **Feature Extraction** converged fastest, accuracy jumped from 73% to 83% in just 2 epochs, confirming EfficientNetB0's pretrained features are highly relevant for food classification
- **Fine-tuning** showed steady improvement across all 10 additional epochs without triggering EarlyStopping
- **Gradual Unfreezing** had the longest training (40 epochs) but the biggest gains came from the first 15 epochs (top + block7 + block6). Later blocks showed diminishing returns
- **Layer-wise LR** converged poorly, validation accuracy fluctuated between 80–86% and EarlyStopping triggered at epoch 11
 
### Generalization
 
| Strategy | Train Accuracy | Val Accuracy | Gap |
|----------|:---:|:---:|:---:|
| Feature Extraction | ~89% | ~88% | ~1% (excellent) |
| Fine-tuning | ~92% | ~89% | ~3% (good) |
| Gradual Unfreezing | ~99% | ~92% | ~7% (moderate overfitting) |
| Layer-wise LR | ~91% | ~81% | ~10% (poor, unstable) |
 
Feature Extraction generalized best (smallest gap) because only 14K parameters were trained limited capacity prevents memorization. Gradual Unfreezing had the highest absolute accuracy but also the largest train/val gap, indicating some overfitting. However, it still achieved the best test accuracy (94.23%), meaning the overfitting did not prevent strong generalization.
 
### Overfitting
 
- **Feature Extraction**: Minimal overfitting the frozen backbone acts as a strong regularizer
- **Fine-tuning**: Mild overfitting more trainable params but controlled by small LR
- **Gradual Unfreezing**: Moderate overfitting in later blocks (train ~99%, val ~92%). The model memorized training data but still generalized well. Could be reduced with stronger augmentation or higher dropout
- **Layer-wise LR**: Severe instability rather than traditional overfitting opening all layers at once without a pretrained head caused noisy optimization. The validation loss increased while training loss decreased, triggering EarlyStopping
 
### Why Gradual Unfreezing Won
 
Gradual Unfreezing outperformed all other strategies because it combines the benefits of controlled training with progressive complexity. Each block gets dedicated epochs to stabilize before new layers are introduced. The learning rate decay ensures deeper (more general) layers are modified minimally. This approach avoids both the limitation of Feature Extraction (frozen backbone) and the instability of Layer-wise LR (everything open at once).
 
---
 
## Bonus: Experiment Tracking
 
- **DAGsHub**: Repository at `ahad-m/my-first-repo` dataset uploaded 
- **MLflow**: All 4 runs tracked with:
  - Parameters: model name, strategy, LR, epochs, batch size, dropout, etc.
  - Metrics: train/val loss and accuracy per epoch, test loss and accuracy
  - Artifacts: training curve plots, comparison charts
  - Models: all 4 models saved with `mlflow.keras.log_model()`
  - Model Registry: best model (Gradual Unfreezing) registered as `food11-efficientnet-best`
 
All experiments are versioned and comparable on the [DAGsHub MLflow dashboard](https://dagshub.com/ahad-m/my-first-repo.mlflow).
 
