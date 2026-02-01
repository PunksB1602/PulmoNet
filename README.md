
# PulmoNet

PulmoNet is a PyTorch-based pipeline for multi-label chest X-ray (CXR) classification and model explainability. It is designed for experiments on the CheXpert dataset and supports multiple convolutional neural network (CNN) backbones. The pipeline includes patient-aware data splitting (train/validation CSVs are assumed to be patient-disjoint; the training script additionally supports patient-level subsampling to prevent leakage during subset experiments), training, evaluation, and Grad-CAM visualizations. PulmoNet aims to provide reproducible and interpretable results for multi-label CXR classification tasks.

## Supported Models

- DenseNet-121
- ResNet-50
- EfficientNet-B0

## Dataset

PulmoNet is configured for the CheXpert-v1.0-small dataset, a large public collection of chest radiographs annotated for multiple thoracic pathologies.
Each image may contain multiple labels, making this a multi-label classification task.

### Target Pathologies

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Pleural Effusion

### Expected Directory Structure

```
data/
└── CheXpert-v1.0-small/
        ├── train.csv
        ├── valid.csv
        ├── train/
        └── valid/
```

## Label Processing

CheXpert labels include:

- 1 → Positive
- 0 → Negative
- -1 → Uncertain
- Blank → Not mentioned

PulmoNet applies the following preprocessing:

- Blank labels → 0
- Uncertain labels (-1) → mapped using a configurable policy:
    - `ones` (default): -1 → 1
    - `zeros`: -1 → 0

## Setup

Clone the repository and navigate into it:

```bash
git clone https://github.com/PunksB1602/PulmoNet.git
cd PulmoNet
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Unix / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

### Configuration

- Optimizer: Adam
- Loss: BCEWithLogitsLoss
- Input resolution: 224 × 224
- Batch size: 32
- Automatic Mixed Precision (AMP): CUDA only
- Metrics: Macro AUROC, Macro PR-AUC
- Reproducibility: deterministic seeding + saved run configuration

All training runs automatically save:

- Best model checkpoint
- Training logs
- Metric curves
- Full run configuration (`run_config.json`)

### Run Training



EfficientNet-B0:

```bash
python -m scripts.train --model efficientnet_b0 --epochs 10 --batch_size 32 --lr 1e-4 --num_workers 2 --seed 42 --amp
```

DenseNet-121:

```bash
python -m scripts.train --model densenet121 --epochs 10 --batch_size 32 --lr 1e-4 --num_workers 2 --seed 42 --amp
```

ResNet-50:

```bash
python -m scripts.train --model resnet50 --epochs 10 --batch_size 32 --lr 1e-4 --num_workers 2 --seed 42 --amp
```

#### Quick Debug Run

To quickly check that the pipeline works end-to-end, you can run a minimal training session using only 1% of the data and a single epoch:



```bash
python -m scripts.train --model efficientnet_b0 --epochs 1 --subset 0.01 --num_workers 2
```

Set --num_workers to the number of CPU cores available for optimal data loading performance (e.g., 2, 4, 8, or higher).

To find your CPU core count in Python:

```python
import os
print(os.cpu_count())
```

Set --num_workers to this value or slightly less (e.g., cpu_count() - 1) for best results.

This is useful for debugging and verifying that your setup is correct before running full experiments.


### Training Artifacts

```
artifacts/<model>/
├── best_model.pth
├── training_log.csv
├── curves.png
└── run_config.json
```

## Subset Experiments

For rapid experimentation, PulmoNet supports training on a fraction of the dataset.

Two modes are available:

- Patient-level (default, recommended): samples patients and includes all their images
- Image-level: samples individual images (for quick debugging)

Example (10% patient-level subset):

```bash
python scripts/train.py --model resnet50 --subset 0.1 --subset_mode patient
```

## Evaluation (Final / Test Results)

After training, evaluate all models using the best checkpoint selected by validation AUROC:

```bash
python scripts/evaluate.py
```

This produces:

- artifacts/final_evaluation_results.csv
- artifacts/final_evaluation_results.md

These tables contain per-label AUROC, PR-AUC, and macro averages and represent the final test results of the project.

## Explainability (Grad-CAM)

Generate Grad-CAM visualizations for trained models:

```bash
python scripts/gradcam.py --model resnet50
```

Options:

Select a specific pathology:

```bash
python scripts/gradcam.py --model resnet50 --class_idx 3
```

Grad-CAM outputs are saved to:

```
artifacts/<model>/
├── gradcam_<label>.png
```

## Results (10% Training Data, 5 Epochs)

Macro AUROC on CheXpert validation set

| Model           | Atelectasis | Cardiomegaly | Consolidation | Edema  | Pleural Effusion | Mean AUROC |
|-----------------|-------------|--------------|---------------|--------|------------------|------------|
| DenseNet-121    | 0.8062      | 0.7960       | 0.8467        | 0.8661 | 0.9137           | 0.8457     |
| ResNet-50       | 0.8189      | 0.7395       | 0.8924        | 0.9116 | 0.9143           | 0.8553     |
| EfficientNet-B0 | 0.8214      | 0.7479       | 0.8586        | 0.9110 | 0.9012           | 0.8480     |

## Notes

- Due to the absence of a separate test split in CheXpert-v1.0-small, validation performance is reported as the final evaluation, following common practice in prior work.
- Curves visualize learning dynamics; reported tables represent final model performance.
- Grad-CAM visualizations are generated from model logits prior to sigmoid activation.
- This project is intended for non-commercial, research-only use.

