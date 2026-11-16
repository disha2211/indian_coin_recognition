# Indian Coin Recognition System

A Flask-based machine learning application for recognizing Indian coin denominations from images. The application provides a workflow for ZIP dataset upload, image preprocessing, Adaptive Feature Enhancement (AFE), reproducible data splitting, model training, evaluation, saved-model inspection, and single-image prediction.

## Features

- Upload and inspect a ZIP dataset; identify denomination folders and report valid, invalid, or corrupt images.
- Resize images to 128 × 128 RGB, normalize pixel values to [0, 1], assign numeric labels, and save NumPy arrays.
- Apply CLAHE contrast enhancement and extract intensity, Sobel-gradient, and Local Binary Pattern (LBP) features.
- Create a stratified 80/20 train/test split using random state 42 and save the exact indices.
- Train Multinomial Naive Bayes, Decision Tree, and Random Forest classifiers using AFE feature vectors.
- Train a proposed DCNN using CLAHE-enhanced grayscale images.
- Evaluate all four models on the same held-out test set.
- Compare accuracy, macro precision, macro recall, and macro F1-score; inspect per-class reports and confusion matrices.
- Predict the denomination of an uploaded JPG, JPEG, or PNG coin image.
- Persist models, preprocessing metadata, AFE configuration, class mapping, split indices, training history, evaluation reports, and plots.

## Technology Stack

| Area | Technologies |
|---|---|
| Web application | Python, Flask, Jinja templates |
| Image processing | OpenCV, Pillow |
| Numerical processing | NumPy |
| Classical machine learning | scikit-learn |
| Deep learning | TensorFlow / Keras |
| Model persistence | Joblib, Keras model format |
| Visualization | Matplotlib, Seaborn |

## Workflow

1. Upload and inspect the ZIP dataset.
2. Preprocess images and generate labels.
3. Run AFE and save numerical feature vectors.
4. Create or load the stratified train/test split.
5. Train the three classical models.
6. Prepare CLAHE-enhanced grayscale inputs and train the DCNN.
7. Evaluate all four models on the same held-out test samples.
8. Compare metrics and inspect classification reports and confusion matrices.
9. Inspect saved models and predict the denomination of a new coin image.

## Adaptive Feature Enhancement (AFE)

AFE is the feature-engineering pipeline used by the classical classifiers. It applies deterministic enhancement and feature extraction to dataset images and prediction images.

### Processing stages

1. **CLAHE contrast enhancement:** Convert normalized RGB to grayscale and apply CLAHE (clip limit 2.0, tile grid 8 × 8).
2. **Enhanced-intensity statistics:** Calculate mean, standard deviation, minimum, maximum, and the 25th, 50th, 75th, and 90th percentiles (8 features).
3. **Sobel-gradient statistics:** Calculate horizontal and vertical Sobel gradients using a 3 × 3 kernel, compute gradient magnitude, normalize it, and calculate the same eight statistics (8 features).
4. **Uniform LBP texture histogram:** Extract Local Binary Pattern texture information using 8 neighboring points and radius 1; produce a normalized 10-bin histogram (10 features).

The final feature vector contains **26 values per image**:

| Feature group | Features |
|---|---:|
| Enhanced-intensity statistics | 8 |
| Sobel-gradient statistics | 8 |
| Uniform LBP histogram | 10 |
| **Total** | **26** |

AFE configuration and metadata are saved with the processed dataset.

### DCNN input clarification

The DCNN uses the CLAHE-enhanced grayscale image (128 × 128 × 1) as its input. It shares the CLAHE enhancement stage with AFE, but it does **not** consume the 26-dimensional classical feature vector. Classical models learn from handcrafted numerical features, while the DCNN learns spatial image features through convolutional layers.

## Models

### Classical classifiers

| Model | Input | Description |
|---|---|---|
| Multinomial Naive Bayes | 26 AFE features | Probabilistic baseline; requires nonnegative features |
| Decision Tree | 26 AFE features | Rule-based classifier |
| Random Forest | 26 AFE features | Ensemble of decision trees |

### Proposed DCNN architecture

The convolutional network uses:

1. Input: 128 × 128 × 1 enhanced grayscale image.
2. Conv2D: 32 filters, 3 × 3 kernel, ReLU, followed by 2 × 2 max pooling.
3. Conv2D: 64 filters, 3 × 3 kernel, ReLU, followed by 2 × 2 max pooling.
4. Conv2D: 128 filters, 3 × 3 kernel, ReLU, followed by 2 × 2 max pooling.
5. Global average pooling.
6. Dense layer: 128 units, ReLU.
7. Dropout: 0.4.
8. Softmax output layer for the coin classes.

Training uses Adam (learning rate 0.001), sparse categorical cross-entropy, batch size 16, and a maximum of 25 epochs. Early stopping and learning-rate reduction monitor validation loss. Class weights are calculated from the training subset.

The validation subset is created from the training indices only. The saved test indices remain held out for final evaluation.

## Evaluation

All four models are evaluated on the same saved test indices. The application reports:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Per-class precision, recall, F1-score, and support
- Confusion matrix

The best model is selected using **macro F1-score**, which gives each class equal weight in the selection metric. Actual performance depends on the dataset and training run; no fixed accuracy is guaranteed.

## Dataset Format

Upload a ZIP archive containing one folder per denomination. Each folder should contain images belonging to that class.

```text
coin_dataset/
├── 1/
│   ├── coin_001.jpg
│   └── coin_002.jpg
├── 2/
│   ├── coin_001.jpg
│   └── ...
├── 5/
│   └── ...
└── 10/
    └── ...
```

The folder names become class labels. Keep the denomination folders directly under the dataset root (or inside one common wrapper folder in the ZIP). At least two classes are required. Use enough images in each class to support stratified splitting and training.

Supported dataset image extensions: JPG, JPEG, PNG, BMP, WEBP. Corrupt or unreadable images are reported during inspection or preprocessing. The single-image prediction form accepts JPG, JPEG, and PNG.

## Installation and Setup

### Prerequisites

- Python (Python 3.11 is a practical choice for this dependency set)
- pip
- Sufficient disk space for extracted images, processed arrays, model files, and evaluation artifacts

TensorFlow and OpenCV may require platform-compatible Python wheels. If installation fails, verify Python and package compatibility for your operating system.

### 1. Clone the repository

```bash
git clone https://github.com/disha2211/indian_coin_recognition.git
cd indian_coin_recognition
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
python run.py
```

Open the local address printed by Flask in your browser (normally http://127.0.0.1:5000/).

## Using the Application

1. Open the dashboard.
2. Open the dataset workspace and upload the ZIP archive.
3. Review detected denomination folders, image counts, and sample previews.
4. Run preprocessing.
5. Run AFE.
6. Generate or load the train/test split.
7. Train each classical model and the proposed DCNN.
8. Run model evaluation after all four models have been trained.
9. Review the metric comparison, classification reports, and confusion matrices.
10. Select a saved model and upload a coin image in the prediction section.

Compatible saved artifacts are reused where supported. Use the force-retrain option in a training form when you explicitly want to train that model again.

## Project Structure

```text
indian_coin_recognition/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── routes/
│   │   └── dataset.py
│   ├── services/
│   │   ├── dataset_service.py
│   │   ├── preprocessing_service.py
│   │   ├── afe_service.py
│   │   ├── split_service.py
│   │   ├── model_training_service.py
│   │   ├── dcnn_service.py
│   │   ├── evaluation_service.py
│   │   ├── prediction_service.py
│   │   ├── model_registry_service.py
│   │   └── state_service.py
│   └── templates/
│       ├── dashboard.html
│       └── dataset.html
├── data/                 # Runtime data and generated artifacts
├── requirements.txt
├── run.py
└── README.md
```

## Saved Artifacts

Artifacts are organized by dataset ID.

| Artifact | Purpose |
|---|---|
| `images.npy` | Preprocessed RGB image array |
| `labels.npy` | Numeric class labels |
| `class_mapping.json` | Mapping between denomination names and numeric labels |
| `preprocessing_metadata.json` | Preprocessing details and counts |
| `afe/features.npy` | 26-dimensional AFE feature matrix |
| `afe/afe_config.json` | AFE methods and parameters |
| `afe/afe_metadata.json` | AFE processing metadata |
| `split/train_indices.npy` | Saved training sample indices |
| `split/test_indices.npy` | Saved test sample indices |
| `models/<dataset_id>/` | Classical model files and metadata |
| `processed/<dataset_id>/dcnn/` | Enhanced DCNN inputs, Keras model, training history, and plots |
| `models/<dataset_id>/evaluations/` | Per-model reports, confusion matrices, and comparison results |
| `artifacts/project_state.json` | Persistent application workflow state |

Generated datasets, models, and runtime artifacts are excluded from Git. Run the application with your own dataset to create them.

## Configuration

Application settings are defined in `app/config.py`.

| Setting | Default |
|---|---:|
| Image size | 128 × 128 |
| Pixel normalization | 0–1 |
| Test split | 20% |
| Random state | 42 |
| Preprocessing batch size | 32 |
| Maximum ZIP upload size | 10 GiB |
| Maximum extracted archive size | 50 GiB |
| Maximum archive entries | 1,000,000 |

Upload and extraction limits can be overridden using the environment variables `MAX_UPLOAD_BYTES`, `MAX_EXTRACTED_BYTES`, and `MAX_ARCHIVE_FILES`. These are disk/archive limits, not RAM allocations. Large datasets still require sufficient disk space and processing time.

## Notes and Limitations

- Model performance depends on image quality, dataset size, class balance, and similarity between training images and real prediction images.
- The current application is intended for local demonstration and academic evaluation. Review deployment security before exposing it publicly.
- The development entry point runs Flask with debug mode enabled. Do not use debug mode in production.
- The Flask secret key is currently a development placeholder in the application factory. Replace it with a securely configured environment value before deployment.
- Keep the original dataset backed up. Generated arrays and trained models are stored locally and are not committed to the repository.

## License

No license file is currently specified in this repository. Add a license before redistributing the project if you intend to grant reuse rights.
