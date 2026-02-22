"""
Training script for Economic Impact Scoring Model

Trains an XGBoost regression model to predict census tract vulnerability scores
based on sea level exposure, economic indicators, workforce composition, and social factors.

Usage:
    python train_economic_model.py                    # Train on all available data
    python train_economic_model.py --test             # Quick test with limited data
    python train_economic_model.py --cv               # Include cross-validation
    python train_economic_model.py --plot             # Generate visualizations

Author: Data4Good Team
Date: February 2026
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.economic_impact_model import (
    train_economic_impact_model,
    cross_validate_economic_model,
)
from api import get_tracts

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_tracts_data(test_mode: bool = False) -> pd.DataFrame:
    """
    Load census tract data from API
    
    Args:
        test_mode: If True, return limited sample
        
    Returns:
        DataFrame of census tract data
    """
    logger.info("Loading census tract data...")
    
    tracts_df = get_tracts()
    
    if test_mode:
        # Use subset for quick testing
        tracts_df = tracts_df.head(10)
        logger.info(f"  Test mode: Using {len(tracts_df)} tracts")
    else:
        logger.info(f"  Loaded {len(tracts_df)} tracts")
    
    return tracts_df


def main():
    parser = argparse.ArgumentParser(
        description='Train Economic Impact Scoring Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_economic_model.py                 # Full training
  python train_economic_model.py --test          # Quick test run
  python train_economic_model.py --cv --plot     # With cross-validation and plots
        """
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: train on small subset for quick validation'
    )
    
    parser.add_argument(
        '--cv',
        action='store_true',
        help='Perform cross-validation after training'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate feature importance and prediction plots'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models/economic_impact_model.pkl',
        help='Path to save trained model (default: models/economic_impact_model.pkl)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving the model'
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        tracts_df = load_tracts_data(test_mode=args.test)
        
        if len(tracts_df) < 10:
            logger.error("❌ ERROR: Not enough tracts for training (need at least 10)")
            logger.error("  Check that census tract data is available")
            sys.exit(1)
        
        # Train model
        save_path = None if args.no_save else Path(args.output)
        
        model, test_metrics = train_economic_impact_model(
            tracts_df=tracts_df,
            housing_df=None,  # Optional: Add if available
            noaa_df=None,     # Optional: Add if available
            workforce_df=None,  # Optional: Add if available
            save_path=save_path,
        )
        
        # Generate plots if requested
        if args.plot:
            logger.info("\nGenerating visualizations...")
            
            plots_dir = Path('models/plots')
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Feature importance
            importance_path = plots_dir / 'economic_impact_feature_importance.png'
            model.plot_feature_importance(save_path=importance_path)
            
            logger.info("  ✓ Plots saved to models/plots/")
        
        # Cross-validation if requested
        if args.cv:
            logger.info("\n")
            cv_results = cross_validate_economic_model(
                tracts_df=tracts_df,
                cv_folds=5,
            )
            
            logger.info("\n✓ Cross-validation complete")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        
        if save_path:
            logger.info(f"\n✓ Model saved to: {save_path}")
            file_size_kb = save_path.stat().st_size / 1024
            logger.info(f"  Model size: {file_size_kb:.1f} KB")
        
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Test predictions:")
        logger.info(f"     python -c \"from src.models.economic_impact_model import EconomicImpactModel; m = EconomicImpactModel.load('{save_path}'); print('Model loaded successfully')\"")
        logger.info(f"  2. Add API endpoint to serve predictions")
        logger.info(f"  3. Update dashboard to display vulnerability scores")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"\n❌ ERROR: Training failed")
        logger.error(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
