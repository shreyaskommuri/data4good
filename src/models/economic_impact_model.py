"""
Economic Impact Scoring - Machine Learning Model

XGBoost regression model to predict census tract vulnerability scores (0-100)
based on sea level exposure, economic indicators, workforce composition, and social factors.

Author: Data4Good Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
import logging
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from .economic_impact_features import EconomicImpactFeatureExtractor, get_feature_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EconomicImpactModel:
    """
    XGBoost Regression model for predicting tract-level economic vulnerability scores
    """
    
    # Feature columns used by the model
    FEATURE_COLUMNS = [
        'flood_zone',
        'high_water_events_annual',
        'coastal_proximity_km',
        'median_income',
        'poverty_rate',
        'housing_pressure_index',
        'housing_affordability_ratio',
        'coastal_jobs_pct',
        'climate_sensitive_jobs_pct',
        'unemployment_rate',
        'job_stability_index',
        'ej_percentile',
        'limited_english_pct',
        'minority_pct',
        'elderly_pct',
        'population',
        'population_density',
    ]
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """
        Initialize the economic impact model
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
            subsample: Fraction of samples per tree
            colsample_bytree: Fraction of features per tree
            random_state: Random seed for reproducibility
        """
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective='reg:squarederror',
            tree_method='hist',
            enable_categorical=True,
        )
        self.feature_names = self.FEATURE_COLUMNS
        self.is_trained = False
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
        verbose: bool = True,
    ):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets (vulnerability scores)
            eval_set: Validation set for monitoring (not used for early stopping in regression)
            verbose: Whether to print training progress
        """
        logger.info("Training Economic Impact Model...")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Features: {len(self.feature_names)}")
        logger.info(f"  Target range: {y_train.min():.1f} - {y_train.max():.1f}")
        
        # Fit without early_stopping_rounds for regression
        self.model.fit(
            X_train[self.feature_names],
            y_train,
            verbose=verbose,
        )
        
        self.is_trained = True
        logger.info("✓ Training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict vulnerability scores
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of predicted vulnerability scores (0-100)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X[self.feature_names])
        
        # Clip predictions to valid range [0, 100]
        predictions = np.clip(predictions, 0, 100)
        
        return predictions
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "Validation"
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Feature dataframe
            y: True vulnerability scores
            dataset_name: Name for logging
            
        Returns:
            Dictionary of metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate percentage of predictions within ±10 points
        within_10 = np.mean(np.abs(y - y_pred) < 10) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'within_10_pct': within_10,
        }
        
        logger.info(f"\n{dataset_name} Set Metrics:")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  MAE: {mae:.2f}")
        logger.info(f"  R²: {r2:.3f}")
        logger.info(f"  Within ±10 points: {within_10:.1f}%")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = None) -> pd.DataFrame:
        """
        Get feature importance rankings
        
        Args:
            top_n: Return only top N features (None = all)
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 15, save_path: Optional[Path] = None):
        """Plot feature importance"""
        importance_df = self.get_feature_importance(top_n=top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance Score')
        plt.title('Economic Impact Model - Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        else:
            plt.show()
    
    def plot_predictions(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None
    ):
        """Plot predicted vs actual vulnerability scores"""
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5, s=50)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # ±10 point bands
        plt.plot([min_val, max_val], [min_val + 10, max_val + 10], 'g--', alpha=0.3, label='±10 points')
        plt.plot([min_val, max_val], [min_val - 10, max_val - 10], 'g--', alpha=0.3)
        
        plt.xlabel('True Vulnerability Score')
        plt.ylabel('Predicted Vulnerability Score')
        plt.title('Economic Impact Model - Predictions vs Actuals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved predictions plot to {save_path}")
        else:
            plt.show()
    
    def save(self, path: Path):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        file_size_kb = path.stat().st_size / 1024
        logger.info(f"✓ Model saved to {path} ({file_size_kb:.1f} KB)")
    
    @classmethod
    def load(cls, path: Path) -> 'EconomicImpactModel':
        """Load model from disk"""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"✓ Model loaded from {path}")
        return model


def train_economic_impact_model(
    tracts_df: pd.DataFrame,
    housing_df: Optional[pd.DataFrame] = None,
    noaa_df: Optional[pd.DataFrame] = None,
    workforce_df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    save_path: Optional[Path] = None,
) -> Tuple[EconomicImpactModel, Dict[str, float]]:
    """
    Complete training pipeline for economic impact model
    
    Args:
        tracts_df: Census tract data
        housing_df: Housing data (optional)
        noaa_df: NOAA water levels (optional)
        workforce_df: Workforce composition (optional)
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed
        save_path: Where to save trained model
        
    Returns:
        Tuple of (trained_model, test_metrics)
    """
    logger.info("="*60)
    logger.info("ECONOMIC IMPACT MODEL TRAINING PIPELINE")
    logger.info("="*60)
    
    # Step 1: Feature extraction
    logger.info("\n[1/5] Extracting features from census tracts...")
    extractor = get_feature_extractor()
    extractor.load_data(tracts_df, housing_df, noaa_df, workforce_df)
    
    X, y = extractor.extract_batch()
    
    logger.info(f"\n  Dataset: {len(X)} tracts")
    logger.info(f"  Features: {len(EconomicImpactModel.FEATURE_COLUMNS)}")
    logger.info(f"  Target: Vulnerability Score (0-100)")
    
    # Step 2: Train/val/test split
    logger.info("\n[2/5] Splitting data...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    logger.info(f"  Train: {len(X_train)} tracts")
    logger.info(f"  Validation: {len(X_val)} tracts")
    logger.info(f"  Test: {len(X_test)} tracts")
    
    # Step 3: Train model
    logger.info("\n[3/5] Training XGBoost regression model...")
    
    model = EconomicImpactModel(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=random_state,
    )
    
    eval_set = [(X_val[model.feature_names], y_val)]
    model.train(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )
    
    # Step 4: Evaluate
    logger.info("\n[4/5] Evaluating model performance...")
    
    train_metrics = model.evaluate(X_train, y_train, "Training")
    val_metrics = model.evaluate(X_val, y_val, "Validation")
    test_metrics = model.evaluate(X_test, y_test, "Test")
    
    # Feature importance
    logger.info("\n  Top 10 Most Important Features:")
    importance_df = model.get_feature_importance(top_n=10)
    for idx, row in importance_df.iterrows():
        logger.info(f"    {row['feature']}: {row['importance']:.3f}")
    
    # Step 5: Save model
    if save_path:
        logger.info(f"\n[5/5] Saving model to {save_path}...")
        model.save(save_path)
    else:
        logger.info("\n[5/5] Skipping model save (no path provided)")
    
    logger.info("\n" + "="*60)
    logger.info("✓ SUCCESS! Economic Impact Model Training Complete")
    logger.info("="*60)
    logger.info(f"\nModel Performance Summary:")
    logger.info(f"  Test RMSE: {test_metrics['rmse']:.2f} points")
    logger.info(f"  Test MAE: {test_metrics['mae']:.2f} points")
    logger.info(f"  Test R²: {test_metrics['r2']:.3f}")
    logger.info(f"  Predictions within ±10 points: {test_metrics['within_10_pct']:.1f}%")
    
    return model, test_metrics


def cross_validate_economic_model(
    tracts_df: pd.DataFrame,
    housing_df: Optional[pd.DataFrame] = None,
    noaa_df: Optional[pd.DataFrame] = None,
    workforce_df: Optional[pd.DataFrame] = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, List[float]]:
    """
    Perform cross-validation on economic impact model
    
    Args:
        tracts_df: Census tract data
        housing_df: Housing data (optional)
        noaa_df: NOAA data (optional)
        workforce_df: Workforce data (optional)
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Dictionary of cross-validation scores
    """
    logger.info("="*60)
    logger.info(f"CROSS-VALIDATION ({cv_folds}-Fold)")
    logger.info("="*60)
    
    # Extract features
    extractor = get_feature_extractor()
    extractor.load_data(tracts_df, housing_df, noaa_df, workforce_df)
    X, y = extractor.extract_batch()
    
    # Create model
    model = EconomicImpactModel(random_state=random_state)
    
    # Cross-validation scoring
    logger.info("\nRunning cross-validation...")
    
    # R² scores
    r2_scores = cross_val_score(
        model.model, X[model.feature_names], y,
        cv=cv_folds, scoring='r2', n_jobs=-1
    )
    
    # Negative MAE (sklearn returns negative for MAE)
    mae_scores = -cross_val_score(
        model.model, X[model.feature_names], y,
        cv=cv_folds, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    
    # Negative RMSE
    rmse_scores = np.sqrt(-cross_val_score(
        model.model, X[model.feature_names], y,
        cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1
    ))
    
    logger.info("\nCross-Validation Results:")
    logger.info(f"  R² scores: {r2_scores}")
    logger.info(f"  Mean R²: {r2_scores.mean():.3f} (+/- {r2_scores.std() * 2:.3f})")
    logger.info(f"\n  MAE scores: {mae_scores}")
    logger.info(f"  Mean MAE: {mae_scores.mean():.2f} (+/- {mae_scores.std() * 2:.2f})")
    logger.info(f"\n  RMSE scores: {rmse_scores}")
    logger.info(f"  Mean RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std() * 2:.2f})")
    
    return {
        'r2_scores': r2_scores.tolist(),
        'mae_scores': mae_scores.tolist(),
        'rmse_scores': rmse_scores.tolist(),
        'mean_r2': float(r2_scores.mean()),
        'mean_mae': float(mae_scores.mean()),
        'mean_rmse': float(rmse_scores.mean()),
    }
