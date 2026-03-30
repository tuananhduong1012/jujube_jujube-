"""utils/download.py – dataset download helpers (Kaggle / Google Drive)."""

import os
import sys
import zipfile
from pathlib import Path


def download_dataset(source: str, dest: str = "./data") -> str:
    """
    Download a dataset and return the extracted root path.

    source formats:
        kaggle:<owner/dataset-slug>   e.g. kaggle:mondejar/jujube-defects
        gdrive:<file-id>              e.g. gdrive:1aBcD2efGH3ijKLmnOPqrstUVwXyz
    """
    os.makedirs(dest, exist_ok=True)

    if source.startswith("kaggle:"):
        return _download_kaggle(source[7:], dest)
    elif source.startswith("gdrive:"):
        return _download_gdrive(source[7:], dest)
    else:
        sys.exit(
            f"ERROR: Unrecognised --download format '{source}'.\n"
            "  Use 'kaggle:<owner/slug>'  or  'gdrive:<file-id>'."
        )


# ─── Kaggle ───────────────────────────────────────────────────────────────────

def _download_kaggle(slug: str, dest: str) -> str:
    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "kagglehub is not installed.  Run:  pip install kagglehub\n"
            "Then set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )

    print(f"  Downloading from Kaggle: {slug} …")
    path = kagglehub.dataset_download(slug)
    print(f"  Downloaded to: {path}")
    return _find_dataset_root(path)


# ─── Google Drive ─────────────────────────────────────────────────────────────

def _download_gdrive(file_id: str, dest: str) -> str:
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is not installed.  Run:  pip install gdown")

    zip_path = os.path.join(dest, f"{file_id}.zip")
    print(f"  Downloading from Google Drive: {file_id} …")
    gdown.download(id=file_id, output=zip_path, quiet=False)

    extract_dir = os.path.join(dest, file_id)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    os.remove(zip_path)
    print(f"  Extracted to: {extract_dir}")
    return _find_dataset_root(extract_dir)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_dataset_root(base: str) -> str:
    """
    Walk down single-child directories to find the first folder that
    contains multiple sub-folders (i.e. the class-level root).
    """
    p = Path(base)
    while True:
        children = [c for c in p.iterdir() if c.is_dir()]
        if len(children) == 1:
            p = children[0]
        else:
            break
    return str(p)
