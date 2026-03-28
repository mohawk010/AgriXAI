import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = Path(r"D:\plant disease dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)")
MODEL_PATH = BASE_DIR / "resnet50_best.keras"
TEST_DIR = BASE_DIR / "test"

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

print("Loading Test Dataset...")
# Shuffle=False is CRITICAL so predictions line up with true labels
raw_test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

def preprocess_for_resnet(image, label):
    return tf.keras.applications.resnet50.preprocess_input(image), label

test_ds = raw_test_ds.map(preprocess_for_resnet, num_parallel_calls=tf.data.AUTOTUNE)

class_names = raw_test_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes.")

print("\nExtracting true labels...")
y_true = np.concatenate([y for x, y in test_ds], axis=0)

print("Loading Model...")
model = tf.keras.models.load_model(str(MODEL_PATH))

print("\nRunning Model Predictions on Test Set (this may take a few minutes)...")
predictions = model.predict(test_ds, verbose=1)
y_pred_probs = predictions
y_pred = np.argmax(predictions, axis=1)

print("\nGenerating Research Artifacts...")

# 1. Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('research_classification_report.csv')
print("-> Saved research_classification_report.csv")

# 2. Confusion Matrix Plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(22, 18))
sns.heatmap(cm, cmap='Blues', annot=False, fmt='g', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Plant Disease Detection', fontsize=18)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('research_confusion_matrix.png', dpi=300)
plt.close()
print("-> Saved research_confusion_matrix.png")

# 3. Overall Metrics
accuracy = report['accuracy']
macro_f1 = report['macro avg']['f1-score']
weighted_f1 = report['weighted avg']['f1-score']
with open('research_overall_metrics.txt', 'w') as f:
    f.write(f"--- Global Performance Metrics ---\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n")
print("-> Saved research_overall_metrics.txt")

# 4. ROC Curves
y_true_bin = label_binarize(y_true, classes=range(num_classes))
plt.figure(figsize=(12, 10))
# Plot just the first 10 classes to avoid a massive spaghetti graph
for i in range(min(10, num_classes)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{class_names[i][:20]} (AUC={roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (Subset of Classes)', fontsize=14)
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('research_roc_curves.png', dpi=300)
plt.close()
print("-> Saved research_roc_curves.png")

print("\nAll research metrics have been successfully generated!")
