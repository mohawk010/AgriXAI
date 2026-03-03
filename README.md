# AgriXAI — Plant Disease Classification

A deep learning project that classifies **38 plant disease categories** across 14 crops using a Custom CNN and ResNet50 Transfer Learning, with Grad-CAM explainability visualizations.

---

## Results

| Metric | Custom CNN | ResNet50 |
|---|---|---|
| Final Train Accuracy | 98.12% | 99.81% |
| Final Val Accuracy | 96.26% | **99.45%** |
| Final Train Loss | 0.0571 | 0.0059 |
| Final Val Loss | 0.1227 | 0.0214 |

ResNet50 (transfer learning, ImageNet pretrained, conv5 block unfrozen) significantly outperforms the custom CNN trained from scratch.

---

## Dataset

**New Plant Diseases Dataset (Augmented)** — available on [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

- **38 classes** across 14 plant species (Apple, Corn, Grape, Tomato, Potato, etc.)
- ~87,000 training images, ~22,000 validation images
- Images are 128×128 (CNN) / 224×224 (ResNet50), RGB

> The dataset images are not included in this repository due to size (~1.3 GB). Download from Kaggle and place the `train/` and `valid/` folders inside `New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/`.

---

## Project Structure

```
AgriXAI/
├── New Plant Diseases Dataset(Augmented)/
│   └── New Plant Diseases Dataset(Augmented)/
│       ├── Train_plant_disease.ipynb   # Main training notebook
│       ├── training_hist.json          # Custom CNN training history
│       ├── resnet_history.json         # ResNet50 training history
│       ├── train/                      # Training images (not in repo)
│       └── valid/                      # Validation images (not in repo)
├── models/
│   └── resnet50.py                     # ResNet50 model definition (PyTorch)
├── scripts/
│   └── compare_models.py               # Model comparison script
├── training/
│   └── train_resnet.py                 # ResNet50 training script
├── utils/
│   ├── dataset.py                      # DataLoader utilities
│   └── metrics.py                      # Evaluation metrics
├── visualization/
│   ├── gradcam.py                      # Grad-CAM implementation
│   └── visualize.py                    # Visualization helpers
├── artifacts/                          # Saved model outputs (not in repo)
├── requirements.txt
└── README.md
```

---

## Models

### Custom CNN (TensorFlow/Keras)
- 5 convolutional blocks (32 → 512 filters), each with 2 Conv2D + MaxPooling layers
- Dropout (0.25 + 0.4), Dense(1500), Softmax(38)
- Input: 128×128 RGB | Optimizer: Adam (lr=0.0001)

### ResNet50 Transfer Learning (TensorFlow/Keras)
- ImageNet pretrained backbone, `conv5` block unfrozen
- GlobalAveragePooling2D → Dropout(0.5) → Dense(38, softmax)
- Input: 224×224 RGB | Optimizer: Adam (lr=0.0001)

### Grad-CAM Explainability
Gradient-weighted Class Activation Maps are generated for ResNet50 predictions, highlighting the regions of the leaf image that influenced each classification decision.

---

## Setup

**Requirements:** Python 3.8+, GPU recommended

```bash
# Clone the repo
git clone https://github.com/mohawk010/AgriXAI.git
cd AgriXAI

# Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# or with Conda:
conda activate tenserflow_env

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the Notebook
Open and run `New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/Train_plant_disease.ipynb` in Jupyter or VS Code. The notebook covers:
1. Data loading & preprocessing
2. Custom CNN training
3. ResNet50 transfer learning training
4. Model evaluation (accuracy, confusion matrix, classification report)
5. Grad-CAM visualizations
6. Model comparison plots

### Train ResNet50 via Script
```bash
python training/train_resnet.py \
  --data-dir "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)" \
  --output-dir artifacts \
  --epochs 10 \
  --batch-size 32
```

### Compare Models
```bash
python scripts/compare_models.py \
  --data-dir "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)" \
  --model1 artifacts/resnet50_best.pth --model1-type resnet50 \
  --model2 artifacts/trained_plant_disease_model.keras --model2-type other \
  --output-dir artifacts
```

---

## Tech Stack

- **TensorFlow / Keras** — CNN & ResNet50 training
- **PyTorch / torchvision** — ResNet50 model module
- **scikit-learn** — Confusion matrix, classification report
- **matplotlib / seaborn** — Visualizations
- **OpenCV** — Image processing for Grad-CAM
- **Jupyter Notebook** — Interactive training environment

---

## Classes

38 disease/healthy categories including:
`Apple___Apple_scab`, `Apple___Black_rot`, `Corn_(maize)___Common_rust_`, `Grape___Black_rot`, `Tomato___Late_blight`, `Tomato___healthy`, and 32 more.

---

## License

This project is for educational and research purposes.
