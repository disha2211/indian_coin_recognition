# Indian Coin Recognition and Prediction System

A Flask-based implementation of the Indian Coin Recognition task specification.

## Current milestone

Module 1: dataset ZIP upload, safe extraction, automatic category discovery, image validation, image counts, and dataset preview metadata.

## Dataset convention

```text
dataset.zip
└── dataset/
    ├── 1_Rupee/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── 2_Rupee/
    └── 10_Rupee/
```

The category names are discovered from folder names; they are not hard-coded.

## Run

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python run.py
```

Open `http://127.0.0.1:5000/dataset/`.
