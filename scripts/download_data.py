"""
scripts/download_data.py — Download the IEEE-CIS Fraud Detection dataset from Kaggle.

Usage:
    python scripts/download_data.py

Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Create API token at https://www.kaggle.com/settings → API → Create New Token
    3. Place kaggle.json at:
       - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
       - Linux/Mac: ~/.kaggle/kaggle.json
    4. Accept competition rules at:
       https://www.kaggle.com/competitions/ieee-fraud-detection/rules
"""

import sys
import os
import zipfile
import shutil
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import DATA_RAW, KAGGLE_COMPETITION, TRANSACTION_FILE, IDENTITY_FILE


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n❌ Kaggle credentials not found!")
        print("\nTo set up Kaggle API access:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section → click 'Create New Token'")
        print("  3. A file 'kaggle.json' will be downloaded")
        print(f"  4. Move it to: {kaggle_json}")
        print("  5. Accept the competition rules at:")
        print("     https://www.kaggle.com/competitions/ieee-fraud-detection/rules")
        print("\nThen re-run: python scripts/download_data.py")
        return False
    return True


def check_already_downloaded() -> bool:
    """Check if raw data files already exist."""
    if TRANSACTION_FILE.exists() and IDENTITY_FILE.exists():
        t_size = TRANSACTION_FILE.stat().st_size / (1024 ** 2)
        i_size = IDENTITY_FILE.stat().st_size / (1024 ** 2)
        print(f"✅ Data already downloaded:")
        print(f"   {TRANSACTION_FILE.name}: {t_size:.1f} MB")
        print(f"   {IDENTITY_FILE.name}: {i_size:.1f} MB")
        return True
    return False


def download_dataset() -> None:
    """Download and extract the IEEE-CIS fraud detection dataset via Kaggle API."""
    try:
        import kaggle
    except ImportError:
        print("❌ kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading '{KAGGLE_COMPETITION}' competition files...")
    print(f"   Target directory: {DATA_RAW}\n")

    try:
        # Download competition files as zip
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(
            competition=KAGGLE_COMPETITION,
            path=str(DATA_RAW),
            quiet=False,
            force=False,
        )
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nCommon fixes:")
        print("  • Make sure you have accepted competition rules at:")
        print("    https://www.kaggle.com/competitions/ieee-fraud-detection/rules")
        print("  • Verify your kaggle.json is valid")
        sys.exit(1)

    # Extract all zip files in data/raw
    print("\n📦 Extracting zip files...")
    for zip_path in DATA_RAW.glob("*.zip"):
        print(f"   Extracting: {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Only extract train files we need
            members = [m for m in zf.namelist() if m.startswith("train_")]
            zf.extractall(path=DATA_RAW, members=members)
        zip_path.unlink()  # Remove zip after extraction

    # Validate
    if TRANSACTION_FILE.exists() and IDENTITY_FILE.exists():
        t_size = TRANSACTION_FILE.stat().st_size / (1024 ** 2)
        i_size = IDENTITY_FILE.stat().st_size / (1024 ** 2)
        print(f"\n✅ Download complete!")
        print(f"   {TRANSACTION_FILE.name}: {t_size:.1f} MB")
        print(f"   {IDENTITY_FILE.name}: {i_size:.1f} MB")
    else:
        print("\n⚠️  Expected files not found after extraction.")
        print("   Files in data/raw:")
        for f in DATA_RAW.iterdir():
            print(f"     {f.name}")
        print("\nPlease manually place train_transaction.csv and train_identity.csv in data/raw/")


def main():
    print("=" * 60)
    print("IEEE-CIS Fraud Detection — Dataset Download")
    print("=" * 60)

    if check_already_downloaded():
        ans = input("\nFiles already exist. Re-download? [y/N]: ").strip().lower()
        if ans != "y":
            print("Skipping download.")
            return

    if not check_kaggle_credentials():
        sys.exit(1)

    download_dataset()


if __name__ == "__main__":
    main()
