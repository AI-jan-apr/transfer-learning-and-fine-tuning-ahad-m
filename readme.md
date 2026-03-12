[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/DtxdB3_i)
## 🧠 Task Overview

You will apply **Transfer Learning** using **EfficientNet** models with two approaches:  
1. **Feature Extraction**  
2. **Fine-tuning**

⚠️ This task **must be completed in Google Colab or a cloud-based environment**. Training deep models like EfficientNet on local machines without GPU/TPU is highly inefficient and may lead to failed or incomplete experiments.



## 📁 Dataset

Dataset is already downloaded and loaded in the notebook. Preprocess as needed for training.



## 🧪 Experiments

### 1️⃣ Feature Extraction  
- freeze all base layers  
- train only the classification head  

### 2️⃣ Fine-tuning  
- unfreeze last layers  
- retrain full or partial base  

You can enhance fine-tuning with these techniques:

- **Unfreeze only last *n* layers**  
  gradually increase trainable layers instead of full base model

- **Gradual unfreezing**  
  unfreeze layers one block at a time across training epochs

- **Layer-wise learning rate decay**  
  assign smaller LR to earlier layers and higher LR to deeper layers

For each:
- document model version  
- include training/validation metrics  
- write your analysis



## 🧬 Bonus (Optional)

- use **DagsHub** to upload and manage dataset in a cloud bucket  
- track all runs using **MLflow**:
  - versioned experiments  
  - parameters, metrics, artifacts  

## 📝 README Must Include:

- experiment summary  
- plots for metrics  
- observations on:
  - feature extract vs fine-tune  
  - generalization, convergence, overfitting 

## 🔗 Helpful Links

- 📚 EfficientNet models in Keras:  
  https://keras.io/api/applications/efficientnet/

- 🎓 Transfer Learning guide (Keras):  
  https://keras.io/guides/transfer_learning/

- 📦 MLflow for experiment tracking:  
  https://www.mlflow.org/docs/latest/index.html

- ☁️ DVC + DagsHub integration:  
  https://dagshub.com/docs/integrations/dvc/

- 🧑‍🍳 How to freeze/unfreeze layers in Keras:  
  https://keras.io/getting_started/faq/#how-can-i-freeze-layers-in-a-model

- 📈 Using callbacks in Keras (e.g. EarlyStopping, ReduceLROnPlateau):  
  https://keras.io/api/callbacks/
