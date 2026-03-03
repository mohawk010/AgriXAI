# Plant Disease Detection — ResNet50 extension

Quick setup and run instructions for the added PyTorch ResNet50, Grad-CAM, and comparison scripts.

Requirements
- Python 3.8+
- GPU recommended for training (optional)

Install dependencies (CPU):

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# For best PyTorch performance or GPU support, install the proper torch wheel per https://pytorch.org/
# Example (CUDA 11.8):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Smoke-check (quick):

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Run training (ResNet50):

```bash
python training/train_resnet.py --data-dir "New Plant Diseases Dataset(Augmented)" --output-dir artifacts --epochs 10 --batch-size 32
```

Generate Grad-CAM visualizations (notebook demo included):

- Open `New Plant Diseases Dataset(Augmented)/Train_plant_disease.ipynb` and run the added ResNet/Grad-CAM demo cells.
- Or run a script that loads `resnet50_best.pth`, instantiates `GradCAM(model, model.layer4[-1].conv3)`, and calls `save_gradcam_visualizations`.

Compare models (size / inference time):

```bash
python scripts/compare_models.py --data-dir "New Plant Diseases Dataset(Augmented)" --model1 artifacts/resnet50_best.pth --model1-type resnet50 --model2 trained_plant_disease_model.keras --model2-type other --output-dir artifacts
```

Notes
- If `pip install -r requirements.txt` fails for `torch`, follow the official install instructions at https://pytorch.org/ to select the correct wheel for your platform and CUDA version.
- Paths in examples assume you run commands from the repository root (`d:\plant disease dataset`). Adjust `--data-dir` to the folder that directly contains `train/` and `valid/`.
