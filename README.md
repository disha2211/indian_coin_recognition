# Indian Coin Recognition and Prediction System

A Flask-based implementation of the Indian Coin Recognition task specification.

## Current milestone

Module 1: ZIP dataset upload, bounded-memory extraction, automatic category discovery, image validation, category counts, and dataset preview metadata.

## Large dataset configuration

Defaults are configurable through environment variables:

- `MAX_UPLOAD_BYTES`: maximum uploaded ZIP size (default: 10 GiB).
- `MAX_EXTRACTED_BYTES`: maximum uncompressed ZIP size (default: 50 GiB).
- `MAX_ARCHIVE_FILES`: maximum number of ZIP entries (default: 1,000,000).

PowerShell example (run before starting Flask):

```powershell
$env:MAX_UPLOAD_BYTES = "10737418240"       # 10 GiB
$env:MAX_EXTRACTED_BYTES = "53687091200"   # 50 GiB
python run.py
```

The ZIP is stored on disk and extracted incrementally in 1 MiB chunks. Image validation is performed one file at a time. The application does not load the complete ZIP or all image arrays into RAM. Ensure the drive has enough free space for the ZIP, extracted images, and later preprocessing artifacts. A compressed 3 GiB ZIP can expand to substantially more than 3 GiB.

For deployments behind a reverse proxy, configure its request-body limit and upload timeout to allow the chosen upload size. The Flask development server is intended for local development, not production hosting.

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

Category names are discovered from folder names; they are not hard-coded.

## Run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Open `http://127.0.0.1:5000/dataset/`.
