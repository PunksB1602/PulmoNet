# PulmoNet

PulmoNet is a PyTorch pipeline for **multi-label chest X-ray (CXR) classification**
and **Grad-CAM–based explainability**. It is designed for experiments on the **CheXpert dataset**
and supports multiple CNN backbones for fair architectural comparison.

Supported models:
- ResNet-50
- DenseNet-121
- EfficientNet-B0

## Dataset

PulmoNet is configured for the **CheXpert-v1.0-small** dataset, a large public collection of chest
radiographs annotated for multiple thoracic pathologies. Each image may contain more than one
label, making this a **multi-label classification task**.

Target pathologies:
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Pleural Effusion

Expected directory structure:

data/
└── CheXpert-v1.0-small/
    ├── train.csv
    ├── valid.csv
    ├── train/
    └── valid/

## Setup

Clone the repository and install dependencies:

git clone https://github.com/PunksB1602/PulmoNet.git  
cd PulmoNet  

python -m venv .venv  
source .venv/bin/activate      # Windows: .venv\Scripts\activate  
pip install -r requirements.txt  

(Optional) Verify dataset structure:
python check_data.py  

## Training
Training configuration:
- Optimizer: Adam
- Loss: BCEWithLogitsLoss
- Input resolution: 224 × 224
- Batch size: 32
- Automatic Mixed Precision (AMP): Enabled


## Training Results (AUROC)

Model-wise performance on the CheXpert validation set:

DenseNet-121  
Atelectasis: 0.8062  
Cardiomegaly: 0.7960  
Consolidation: 0.8467  
Edema: 0.8661  
Pleural Effusion: 0.9137  
Mean AUROC: 0.8457  

ResNet-50  
Atelectasis: 0.8189  
Cardiomegaly: 0.7395  
Consolidation: 0.8924  
Edema: 0.9116  
Pleural Effusion: 0.9143  
Mean AUROC: 0.8553  

EfficientNet-B0  
Atelectasis: 0.8214  
Cardiomegaly: 0.7479  
Consolidation: 0.8586  
Edema: 0.9110  
Pleural Effusion: 0.9012  
Mean AUROC: 0.8480  

## Grad-CAM Explainability

Generate Grad-CAM visualizations:

python gradcam_resnet.py --model outputs/resnet50/best_model.pth --image path/to/xray.jpg  

Equivalent scripts are available for DenseNet-121 and EfficientNet-B0.



## License

See the LICENSE file for details.
