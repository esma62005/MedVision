# 🩻 MedVision – Bone Fracture Detection

MedVision is a Flask web application that analyzes an X-ray image and predicts whether the bone is **Normal** or **Fractured**, using a deep learning model (`.h5`) built with TensorFlow / Keras.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is **not** a medical device and must not replace the diagnosis of a qualified doctor.

---

## 🧠 The Model (`.h5`)

The repository contains two trained Keras models saved in HDF5 format:

| File | Architecture | Classification head | Optimizer | Used by the app |
|------|--------------|---------------------|-----------|-----------------|
| `best_model.h5` | ResNet50 backbone | Dense(256, ReLU) → Dense(1, Sigmoid) | Adam (lr = 1e-6) | ✅ Yes |
| `bone_fracture_model.h5` | ResNet50 backbone | Flatten → Dense(128, ReLU) → Dense(1, Sigmoid) | Adam (lr = 1e-4) | Experimental |

**Details**

- **Type:** binary image classification (Normal vs. Fractured), transfer learning on a ResNet50 backbone.
- **Input:** RGB image resized to `224 × 224 × 3`, pixel values scaled to `[0, 1]` (divided by 255).
- **Output:** one sigmoid neuron = probability that the bone is fractured.
- **Loss:** `binary_crossentropy`.
- **Metrics (best_model):** accuracy, precision, recall.
- **Decision rule:** probability `< 0.60` → **Normal**, otherwise → **Fractured**. The confidence shown is the probability of the predicted class.
- **Format:** Keras 3 `.h5` file, loaded with `tf.keras.models.load_model()`.

### Load the model

```python
from tensorflow.keras.models import load_model

model = load_model("best_model.h5")
```

### Predict on one image

```python
import tensorflow as tf

img = tf.keras.preprocessing.image.load_img("xray.jpg", target_size=(224, 224))
arr = tf.keras.preprocessing.image.img_to_array(img)
arr = tf.expand_dims(arr, 0) / 255.0

p = float(model.predict(arr)[0][0])
label = "Fractured" if p >= 0.60 else "Normal"
print(label, round(p if p >= 0.60 else 1 - p, 2))
```

---

## ✨ Features

- Upload an X-ray image from the browser
- Instant prediction (Normal / Fractured) with a confidence score
- Simple and modern dark UI
- Lightweight Flask backend

## 🗂️ Project Structure

```
MedVision/
├── app.py                  # Flask app + prediction logic
├── TEMPLATES/index.html    # Web page (rename to "templates" on Linux/macOS)
├── static/uploads/         # Uploaded images
├── best_model.h5           # Main trained model (Git LFS)
├── bone_fracture_model.h5  # Alternative model (Git LFS)
├── requirements.txt
└── README.md
```

> On Linux/macOS Flask looks for a folder named `templates` (lowercase). Rename `TEMPLATES` → `templates` if you get a *TemplateNotFound* error.

## 🚀 Getting Started

```bash
# 1. Clone (Git LFS is needed to download the .h5 models)
git lfs install
git clone https://github.com/esma62005/MedVison.git
cd MedVison

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## 🛠️ Tech Stack

Python · Flask · TensorFlow / Keras · ResNet50 · HTML / CSS / JavaScript

## 📌 Notes

- The model expects X-ray style images; other image types give meaningless results.
- Set `debug=False` (already the default) before deploying.

## 👤 Author

**esma62005**
