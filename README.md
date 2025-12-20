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
```bash
data/
└── CheXpert-v1.0-small/
    ├── train.csv
    ├── valid.csv
    ├── train/
    └── valid/
```

## Setup

Clone the repository and navigate into it:

```bash
git clone https://github.com/PunksB1602/PulmoNet.git
cd PulmoNet
```

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training
Training configuration:
- Optimizer: Adam
- Loss: BCEWithLogitsLoss
- Input resolution: 224 × 224
- Batch size: 32
- Automatic Mixed Precision (AMP): Enabled


## Training Results (AUROC)

| Model          | Atelectasis | Cardiomegaly | Consolidation | Edema  | Pleural Effusion | Mean AUROC |
|----------------|-------------|--------------|---------------|--------|------------------|------------|
| DenseNet-121   | 0.8062      | 0.7960       | 0.8467        | 0.8661 | 0.9137           | 0.8457     |
| ResNet-50      | 0.8189      | 0.7395       | 0.8924        | 0.9116 | 0.9143           | 0.8553     |
| EfficientNet-B0| 0.8214      | 0.7479       | 0.8586        | 0.9110 | 0.9012           | 0.8480     |




## License

See the LICENSE file for details.
