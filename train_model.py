#!/usr/bin/env python3
"""
Train Displacement Risk Model - Quick Start
-------------------------------------------
Train XGBoost model to predict worker displacement risk.

Usage:
    python train_model.py [--records N] [--test]

Examples:
    python train_model.py                    # Train on full dataset
    python train_model.py --records 1000     # Train on 1000 workers
    python train_model.py --test             # Test mode (100 workers)
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.displacement_model import train_displacement_model, cross_validate_model


def main():
    parser = argparse.ArgumentParser(description='Train displacement risk model')
    parser.add_argument('--records', type=int, default=None,
                        help='Maximum number of worker records to use (default: all)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: use only 100 records for quick testing')
    parser.add_argument('--no-cv', action='store_true',
                        help='Skip cross-validation')
    parser.add_argument('--output', type=str, default='models/displacement_risk_model.pkl',
                        help='Output path for trained model')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to Live Data directory (default: drive-download-20260221T181111Z-3-001)')
    
    args = parser.parse_args()
    
    # Configuration - use the same directory as api.py
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    else:
        # Try the actual directory name first
        DATA_DIR = Path(__file__).parent / "drive-download-20260221T181111Z-3-001"
        # Fallback to data/live_data if it doesn't exist
        if not DATA_DIR.exists():
            DATA_DIR = Path(__file__).parent / "data" / "live_data"
    
    MODEL_PATH = Path(__file__).parent / args.output
    
    # Check data directory
    if not DATA_DIR.exists():
        print(f"❌ Error: Data directory not found: {DATA_DIR}")
        print("\nPlease ensure Live Data Technologies JSONL files are in one of:")
        print(f"  {Path(__file__).parent / 'drive-download-20260221T181111Z-3-001'}/")
        print(f"  {Path(__file__).parent / 'data' / 'live_data'}/")
        print("\nOr specify a custom path with --data-dir")
        print("\nExpected files:")
        print("  *.jsonl.gz (compressed worker records)")
        sys.exit(1)
    
    # Check if data files exist
    data_files = list(DATA_DIR.glob("*.jsonl.gz"))
    if not data_files:
        print(f"❌ Error: No JSONL.GZ files found in {DATA_DIR}")
        sys.exit(1)
    
    print(f"✓ Found {len(data_files)} data file(s)")
    
    # Determine max records
    max_records = args.records
    if args.test:
        max_records = 100
        print("\n⚠️  TEST MODE: Using only 100 records for quick validation")
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING DISPLACEMENT RISK MODEL")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Max records:    {max_records or 'all'}")
    print(f"Output:         {MODEL_PATH}")
    print("="*60 + "\n")
    
    try:
        model = train_displacement_model(
            data_path=str(DATA_DIR),
            max_records=max_records,
            save_path=str(MODEL_PATH)
        )
        
        # Cross-validation (optional)
        if not args.no_cv and not args.test:
            print("\n\n" + "="*60)
            print("RUNNING CROSS-VALIDATION")
            print("="*60)
            cross_validate_model(
                data_path=str(DATA_DIR),
                max_records=max_records,
                n_folds=5
            )
        
        print("\n" + "="*60)
        print("✓ SUCCESS!")
        print("="*60)
        print(f"\nModel saved to: {MODEL_PATH}")
        print(f"Model size:     {MODEL_PATH.stat().st_size / 1024:.1f} KB")
        
        print("\nNext steps:")
        print("  1. Test predictions:")
        print("     python -c \"from src.models import get_predictor; p = get_predictor('models/displacement_risk_model.pkl'); print(p.get_model_info())\"")
        print("\n  2. Add API endpoint (see docs/DISPLACEMENT_MODEL.md)")
        print("\n  3. Update dashboard to show displacement risk")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
