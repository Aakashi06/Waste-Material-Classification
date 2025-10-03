import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "TrashType_Image_Dataset")
ARTIFACT_DIR = os.path.join(BASE_DIR, "models")
IMAGE_SIZE = (64, 64)

# Collect image paths and labels
classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
image_paths = []
labels = []
for label in classes:
	folder = os.path.join(DATA_DIR, label)
	for fname in os.listdir(folder):
		if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
			image_paths.append(os.path.join(folder, fname))
			labels.append(label)

data = pd.DataFrame({"path": image_paths, "label": labels})
print(f"Found {len(data)} images across {len(classes)} classes: {classes}")

# Load and resize
X_list = []
y_list = []
for _, row in data.iterrows():
	try:
		img = Image.open(row["path"]).convert("RGB")
		img = img.resize(IMAGE_SIZE)
		arr = np.array(img, dtype=np.uint8)
		X_list.append(arr)
		y_list.append(row["label"])
	except Exception as e:
		print("Skip:", row["path"], "-", e)

X = np.stack(X_list, axis=0)
y = np.array(y_list)

# Encode and split
le = LabelEncoder()
y_enc = le.fit_transform(y)
X_flat = X.reshape(X.shape[0], -1).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
	X_flat, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Scale and train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Eval
acc = (model.score(X_test_scaled, y_test))
print(f"Validation accuracy: {acc:.4f}")

# Save artifacts
os.makedirs(ARTIFACT_DIR, exist_ok=True)
artifacts = {
	"model": model,
	"scaler": scaler,
	"label_classes": list(le.classes_),
	"image_size": IMAGE_SIZE,
}
with open(os.path.join(ARTIFACT_DIR, "waste_classifier.pkl"), "wb") as f:
	pickle.dump(artifacts, f)
print("Saved:", os.path.join(ARTIFACT_DIR, "waste_classifier.pkl")) 