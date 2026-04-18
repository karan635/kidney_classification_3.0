# Kidney Disease Classification using CNN (VGG16)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.2.2-blue)
![DVC](https://img.shields.io/badge/DVC-Enabled-green)
![Flask](https://img.shields.io/badge/Flask-API-lightgrey)

A deep learning project for classifying kidney diseases from medical images using a fine-tuned **VGG16** Convolutional Neural Network. The project follows a modular MLOps pipeline with experiment tracking via **MLflow** and data/pipeline versioning via **DVC**.

---

## 📌 Project Overview

This project classifies kidney CT scan images into categories (e.g., Normal vs. Tumor) using transfer learning on the VGG16 architecture. The end-to-end pipeline covers data ingestion, model preparation, training, evaluation, and deployment via a Flask API.

---

## 🗂️ Project Structure

```
Kidney_classification/
├── src/
│   └── cnnClassifier/
│       ├── components/         # Core logic: data ingestion, model training, evaluation
│       ├── config/             # Configuration manager
│       ├── constants/          # Project-wide constants
│       ├── entity/             # Dataclasses for config entities
│       ├── pipeline/           # Stage-wise pipeline scripts
│       └── utils/              # Utility functions
├── config/
│   └── config.yaml             # Path and artifact configurations
├── research/
│   └── trials.ipynb            # Experimental notebooks
├── templates/
│   └── index.html              # Flask web UI
├── main.py                     # Entry point to run all pipeline stages
├── params.yaml                 # Hyperparameters
├── dvc.yaml                    # DVC pipeline stages
├── setup.py                    # Package setup
└── requirements.txt            # Dependencies
```

---

## ⚙️ Pipeline Stages

The project runs as a 4-stage sequential pipeline defined in `main.py`:

| Stage | Description |
|-------|-------------|
| **Stage 1** | Data Ingestion — Downloads and prepares the dataset |
| **Stage 2** | Prepare Base Model — Loads VGG16 with ImageNet weights, modifies top layers |
| **Stage 3** | Model Training — Fine-tunes the model on kidney CT scan data |
| **Stage 4** | Model Evaluation — Evaluates performance and logs metrics to MLflow |

---

## 🧠 Model Architecture

- **Base Model:** VGG16 (pretrained on ImageNet, `include_top=False`)
- **Input Shape:** `224 × 224 × 3`
- **Output Classes:** 2 (e.g., Normal, Tumor)
- **Top Layers:** Custom fully connected head added for binary classification

---

## 🔧 Hyperparameters (`params.yaml`)

```yaml
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 1
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.01
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/karan635/Kidney_classification-
cd Kidney_classification-
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize the Project Structure

```bash
python template.py
```

---

## 🚀 Running the Pipeline

To run the complete MLOps pipeline (all 4 stages):

```bash
python main.py
```

To run individual DVC stages:

```bash
dvc repro
```

---

## 📊 Experiment Tracking with MLflow

Metrics and parameters are logged automatically during the evaluation stage.

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser to view experiments.

---

## 🌐 Flask Web Application

Start the prediction API:

```bash
python app.py
```

The app serves a web UI at `http://localhost:8080` where you can upload CT scan images for prediction.

---

## 📦 Dependencies

| Package | Version |
|---------|---------|
| TensorFlow | 2.12.0 |
| MLflow | 2.2.2 |
| DVC | latest |
| Flask | latest |
| pandas | latest |
| numpy | latest |
| python-box | 6.0.2 |
| ensure | 1.0.2 |

---

## 👤 Author

**Karan Parwani**  
📧 karanparwani9904@gmail.com  
🔗 [GitHub: karan635](https://github.com/karan635)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
